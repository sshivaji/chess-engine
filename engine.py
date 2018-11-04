import sys
import threading
import cmd
import chess
from chess import polyglot
import tables
import os
import glob

# DGT

from pydgt import DGTBoard
from pydgt import FEN
from pydgt import CLOCK_BUTTON_PRESSED
from pydgt import CLOCK_LEVER
from pydgt import CLOCK_ACK
from pydgt import scan as dgt_port_scan
from threading import Thread, RLock


## Some code adapted from https://github.com/alexsyrom/chess-engine

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

logfile = open(os.path.join(__location__, 'input.log'), 'w')

ENGINE_NAME = 'DGT UCI chess engine'
AUTHOR_NAME = 'Shivkumar Shivaji'
ENGINE_PLAY = "engine_play"

def scan():
   # scan for available ports. return a list of device names.
    return glob.glob('/dev/cu.usb*') + glob.glob('/dev/tty.DGT*') + glob.glob('/dev/ttyACM*')


class KThread(Thread):
    """A subclass of threading.Thread, with a kill()
  method."""
    def __init__(self, *args, **keywords):
        Thread.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        """Start the thread."""
        self.__run_backup = self.run
        self.run = self.__run      # Force the Thread to install our trace.
        Thread.start(self)

    def __run(self):
        """Hacked run function, which installs the
    trace."""
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, why, arg):
        if why == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, why, arg):
        if self.killed:
            if why == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True


class Analyzer(threading.Thread):
    MIN_VALUE = -10 * tables.piece[chess.KING]

    BETA = tables.piece[chess.ROOK]
    ALPHA = -BETA

    MAX_ITER = 2
    MULTIPLIER = 4

    MAX_NEGAMAX_ITER = 2
    NEGAMAX_DIVISOR = 3

    def set_default_values(self):
        self.infinite = False
        self.possible_first_moves = set()
        self.max_depth = 3
        self.number_of_nodes = 100

    def __init__(self, call_if_ready, call_to_inform, opening_book):
        super(Analyzer, self).__init__()
        if opening_book:
            self.opening_book = polyglot.open_reader(opening_book)
        else:
            self.opening_book = None
        self.debug = False
        self.set_default_values()
        self.board = chess.Board()

        self.is_working = threading.Event()
        self.is_working.clear()
        self.is_conscious = threading.Condition()
        self.termination = threading.Event()
        self.termination.clear()

        self._call_if_ready = call_if_ready
        self._call_to_inform = call_to_inform
        self._bestmove = chess.Move.null()

    @property
    def bestmove(self):
        return self._bestmove

    class Communicant:
        def __call__(self, func):
            def wrap(instance, *args, **kwargs):
                if instance.termination.is_set():
                    sys.exit()
                with instance.is_conscious:
                    instance.is_conscious.notify()
                result = func(instance, *args, **kwargs)
                with instance.is_conscious:
                    instance.is_conscious.notify()
                if instance.termination.is_set():
                    sys.exit()
                return result
            return wrap

    @property
    def number_of_pieces(self):
        number = sum(1 for square in chess.SQUARES
                     if self.board.piece_at(square))
        return number

    def evaluate_material_position(self, phase, color, pieces):
        value = 0
        for piece in pieces:
            squares = self.board.pieces(piece, color)
            for square in squares:
                value += tables.piece_square[phase][color][piece][square]
        return value

    def evaluate_material(self, color):
        value = 0
        for piece in chess.PIECE_TYPES:
            squares = self.board.pieces(piece, color)
            value += len(squares) * tables.piece[piece]
        return value

    def evaluate(self):
        if self.board.is_checkmate():
            return self.MIN_VALUE
        if self.board.is_stalemate():
            return 0

        colors = list(map(int, chess.COLORS))

        values = [0 for i in tables.PHASES]
        phase = tables.OPENING
        pieces = list(range(1, 6))  # pieces without king
        for color in colors:
            values[phase] += (self.evaluate_material_position
                              (phase, color, pieces)
                              *
                              (-1 + 2 * color))
        values[tables.ENDING] = values[tables.OPENING]
        for phase in tables.PHASES:
            for color in colors:
                values[phase] += (self.evaluate_material_position
                                  (phase, color, (chess.KING,))
                                  *
                                  (-1 + 2 * color))

        material = [0 for i in colors]
        for color in colors:
            material[color] = self.evaluate_material(color)
        material_sum = sum(material)

        for color in colors:
            for phase in tables.PHASES:
                values[phase] += material[color] * (-1 + 2 * color)

        value = ((values[tables.OPENING] * material_sum +
                  values[tables.ENDING] * (tables.PIECE_SUM - material_sum))
                 // tables.PIECE_SUM)

        if self.board.turn == chess.BLACK:
            value *= -1

        return value

    def moves(self, depth):
        if depth == 0 and self.possible_first_moves:
            for move in self.board.legal_moves:
                if move in self.possible_first_moves:
                    yield move
        else:
            for move in self.board.legal_moves:
                yield move

    def inner_negamax(self, depth, alpha, beta):
        best_value = alpha

        for move in self.moves(depth):
            if self.debug:
                self._call_to_inform('currmove {}'.format(move.uci()))

            self.board.push(move)
            value = -self.negamax(depth+1, -beta, -best_value)

            if self.debug:
                self._call_to_inform('string value {}'.format(value))

            self.board.pop()

            if value >= beta:
                if depth == 0:
                    self._bestmove = move
                return beta
            elif value > best_value:
                best_value = value
                if depth == 0:
                    self._bestmove = move
            elif depth == 0 and not bool(self._bestmove):
                self._bestmove = move

        return best_value

    @Communicant()
    def negamax(self, depth, alpha, beta):
        if depth == self.max_depth or not self.is_working.is_set():
            return self.evaluate()

        if self.debug:
            self._call_to_inform('depth {}'.format(depth))
            self._call_to_inform('string alpha {} beta {}'.format(alpha, beta))

        value = alpha

        left_borders = [beta - (beta - alpha) // self.NEGAMAX_DIVISOR ** i
                        for i in range(self.MAX_NEGAMAX_ITER, -1, -1)]
        for left in left_borders:
            value = self.inner_negamax(depth, left, beta)
            if value > left:
                break

        return value

    def run(self):
        while self.is_working.wait():
            if self.termination.is_set():
                sys.exit()
            self._bestmove = chess.Move.null()

            try:
                if not self.possible_first_moves:
                    entry = self.opening_book.find(self.board)
                    self._bestmove = entry.move()
                else:
                    for entry in self.opening_book.find_all(self.board):
                        move = entry.move()
                        if move in self.possible_first_moves:
                            self._bestmove = move
                            break
            except:
                pass

            if not bool(self._bestmove):
                middle = self.evaluate()
                alpha = self.ALPHA
                beta = self.BETA
                for i in range(self.MAX_ITER):
                    value = self.negamax(depth=0,
                                         alpha=middle+alpha,
                                         beta=middle+beta)
                    if value >= middle + beta:
                        beta *= self.MULTIPLIER
                    elif value <= middle + alpha:
                        alpha *= self.MULTIPLIER
                    else:
                        break
                self._call_to_inform('pv score cp {}'.format(value))
            else:
                self._call_to_inform('string opening')
            if not self.infinite:
                self._call_if_ready()
            self.set_default_values()
            self.is_working.clear()


class EngineShell(cmd.Cmd):
    intro = ''
    prompt = ''
    file = None

    opening_book_list = ['gm2001',
                         'komodo',
                         'Human']
    opening_book = 'Human'
    opening_dir = 'opening'
    opening_book_extension = '.bin'

    go_parameter_list = ['infinite', 'searchmoves', 'depth', 'nodes']

    def __init__(self):
        # super(EngineShell, self).__init__()
        # super(self).__init__()
        cmd.Cmd.__init__(self)
        self.postinitialized = False
        self.dgt_fen = None
        self.computer_move_FEN_reached = False
        self.mode = ENGINE_PLAY
        self.bestmove = None

    def discover_usb_devices(self):
        for port in scan():
            # if port.startswith("/dev/tty.DGT"):
            if port.startswith("/dev/cu.usbmodem"):
                device = port
                print("info string device : {0}".format(device))
                return device
        # cu.DGT_BT_21265 - SPP

    def discover_bluetooth_devices(self, duration=15):
        import bluetooth
        print("info string importing bluetooth")

        nearby_devices = bluetooth.discover_devices(lookup_names=True, duration=duration)
        print("info string found %d devices" % len(nearby_devices))

        for addr, name in nearby_devices:
            print("info string   %s - %s" % (addr, name))
            # return nearby_devices
            if name.startswith("DGT_"):
                self.dgt_device = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.dgt_device.connect(addr, 1)
                print("info string Finished")

    def try_dgt_legal_moves(self, from_fen, to_fen):
        to_fen_first_tok = to_fen.split()[0]
        temp_board = chess.Board(fen=from_fen)

        for m in temp_board.legal_moves:
            temp_board2 = chess.Board(fen=from_fen)
            # print("move: {}".format(m))
            temp_board2.push(m)
            cur_fen = temp_board2.fen()
            cur_fen_first_tok = str(cur_fen).split()[0]
            #            print "cur_token:{0}".format(cur_fen_first_tok)
            #            print "to_token:{0}".format(to_fen_first_tok)
            if cur_fen_first_tok == to_fen_first_tok:
                self.dgt_fen = to_fen
                # print("info string Move received is : {}".format(m))
                self.bestmove = str(m)
                self.output_bestmove()
                # self.process_move(move=str(m))
                return True

    def dgt_probe(self, attr, *args):
        if attr.type == FEN:
            new_dgt_fen = attr.message
            #            print "length of new dgt fen: {0}".format(len(new_dgt_fen))
            #            print "new_dgt_fen just obtained: {0}".format(new_dgt_fen)
            if self.dgt_fen and new_dgt_fen:
                if new_dgt_fen != self.dgt_fen:
                    if self.mode == ENGINE_PLAY:
                        self.computer_move_FEN_reached = False

                    if not self.try_dgt_legal_moves(self.analyzer.board.fen(), new_dgt_fen):
                        dgt_fen_start = new_dgt_fen.split()[0]
                        curr_fen_start = self.analyzer.board.fen().split()[0]
                        if curr_fen_start == dgt_fen_start and self.mode == ENGINE_PLAY:
                            self.computer_move_FEN_reached = True

                    #     if self.chessboard.parent:
                    #         prev_fen_start = self.chessboard.parent.board().fen().split()[0]
                    #         if dgt_fen_start == prev_fen_start:
                    #             self.back('dgt')
                    # if self.engine_mode != ENGINE_PLAY and self.engine_mode != ENGINE_ANALYSIS:
                    #     if self.lcd:
                    #         self.write_lcd_prev_move()

            elif new_dgt_fen:
                self.dgt_fen = new_dgt_fen
        # if attr.type == CLOCK_BUTTON_PRESSED:
        #     print("Clock button {0} pressed".format(attr.message))
        #     e = ButtonEvent(attr.message)
        #     self.dgt_button_event(e)
        # if attr.type == CLOCK_ACK:
        #     self.clock_ack_queue.put('ack')
        #     print
        #     "Clock ACK Received"
        # if attr.type == CLOCK_LEVER:
        #     if self.clock_lever != attr.message:
        #         if self.clock_lever:
        #             # not first clock level read
        #             # print "clock level changed to {0}!".format(attr.message)
        #             e = ButtonEvent(5)
        #             self.dgt_button_event(e)
        #
        #         self.clock_lever = attr.message

    def poll_dgt(self):
        self.dgt_thread = KThread(target=self.dgtnix.poll)
        self.dgt_thread.daemon = True
        self.dgt_thread.start()

    def dgt_board_connect(self, device):
        self.device=""
        self.dgtnix = DGTBoard(device)
        # self.dgtnix.subscribe(self.dgt_probe)
        # poll_dgt()
        self.dgtnix.subscribe(self.dgt_probe)
        self.poll_dgt()
        # sleep(1)
        self.dgtnix.test_for_dgt_clock()
        # p
        # if self.dgtnix.dgt_clock:
        #     print ("Found DGT Clock")
        #     self.dgt_clock_ack_thread()
        # else:
        #     print ("No DGT Clock found")
        self.dgtnix.get_board()

        if not self.dgtnix:
            print ("info strong Unable to connect to the device on {0}".format(self.device))
        else:
            print("info string The board was found")
            self.dgt_connected = True

    def postinit(self):
        opening_book = self.opening_book + self.opening_book_extension
        opening_book = os.path.join(self.opening_dir, opening_book)
        self.analyzer = Analyzer(
            self.output_bestmove,
            self.output_info,
            os.path.join(__location__, opening_book))
        self.analyzer.start()
        device = self.discover_usb_devices()
        if device:
            self.dgt_board_connect(device)
        # self.discover_bluetooth_devices()
        self.postinitialized = True

    def do_uci(self, arg):
        print('id name {}'.format(ENGINE_NAME) )
        print('id author {}'.format(AUTHOR_NAME))

        # for book in self.opening_book_list:
        #     print('var {}'.format(book))

        print('option name OpeningBook type combo default {} {}'.format(self.opening_book, ' var '.join(self.opening_book_list)))

        # print()
        print('uciok')
        self.postinit()

    def do_debug(self, arg):
        arg = arg.split()
        if arg:
            arg = arg[0]
        else:
            return
        if arg == 'on':
            self.analyzer.debug = True
        elif arg == 'off':
            self.analyzer.debug = False

    def do_isready(self, arg):
        if not self.postinitialized:
            self.postinit()
        if self.analyzer.is_working.is_set():
            with self.analyzer.is_conscious:
                self.analyzer.is_conscious.wait()
        print('readyok')

    def do_setoption(self, arg):
        arg = arg.split()
        try:
            if arg[0] != 'name':
                return
            arg.pop(0)
            if (arg[0] == 'OpeningBook' and
                    arg[1] == 'value' and
                    arg[2] in self.opening_book_list):
                self.opening_book = arg[2]
        except:
            pass

    def do_ucinewgame(self, arg):
        pass

    def do_position(self, arg):
        arg = arg.split()
        if not arg:
            return
        if self.analyzer.is_working.is_set():
            '''
                something strange
                according to the protocol I should ignore it
                *if I ignore it, maybe it will go away*
            '''
            return
        if arg[0] == 'fen' and len(arg) >= 7:
            self.analyzer.board.set_fen(' '.join(arg[1:7]))
            del arg[:7]
        else:
            if arg[0] == 'startpos':
                arg.pop(0)
            self.analyzer.board.reset()
        if arg and arg[0] == 'moves':
            for move in arg[1:]:
                self.analyzer.board.push_uci(move)

    def do_go(self, arg):
        print("info string go called")
        # self.output_bestmove()
        # arg = arg.split()
        # for parameter in self.go_parameter_list:
        #     try:
        #         index = arg.index(parameter)
        #     except:
        #         pass
        #     else:
        #         getattr(self, 'go_' + arg[index])(arg[index + 1:])
        # try:
        #     index = arg.index('movetime')
        #     time = float(arg[index + 1]) / 1000
        # except:
        #     pass
        # else:
        #     self.stop_timer = threading.Timer(time, self.do_stop)
        #     self.stop_timer.start()
        # self.analyzer.is_working.set()

    def do_stop(self, arg=None):
        if hasattr(self, 'stop_timer'):
            self.stop_timer.cancel()
        if self.analyzer.is_working.is_set():
            self.analyzer.is_working.clear()
        else:
            self.output_bestmove()

    def do_quit(self, arg):
        if hasattr(self, 'analyzer'):
            self.analyzer.termination.set()
            self.analyzer.is_working.set()
            self.analyzer.join()
        sys.exit()

    def output_bestmove(self):
        # print('bestmove: {}'.format(self.analyzer.bestmove.uci()))
        print('bestmove {}'.format(self.bestmove))


              # file=self.stdout, flush=True)

    def output_info(self, info_string):
        print('info {}'.format(info_string))
              # file=self.stdout, flush=True)

    def go_infinite(self, arg):
        self.analyzer.infinite = True

    def go_searchmoves(self, arg):
        self.analyzer.possible_first_moves = set()
        for uci_move in arg:
            try:
                move = chess.Move.from_uci(uci_move)
            except:
                break
            else:
                self.analyzer.possible_first_moves.add(move)

    def go_depth(self, arg):
        if not self.analyzer.debug:
            return
        try:
            depth = int(arg[0])
        except:
            pass
        else:
            self.analyzer.max_depth = depth

    def go_nodes(self, arg):
        try:
            number_of_nodes = int(arg[0])
        except:
            pass
        else:
            self.analyzer.depth = number_of_nodes

    def default(self, arg):
        pass

    def precmd(self, line):
        print(line)
        return line

    def postcmd(self, stop, line):
        self.stdout.flush()
        return stop


if __name__ == '__main__':
    # print('new start')
    EngineShell().cmdloop()
