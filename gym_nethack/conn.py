import os, subprocess, signal
from sys import platform

import zmq

###############
# Directories #
###############

if platform == "linux" or platform == "linux2":
    nethack_path = "../nh/install/games/lib/nethackdir/nethack"
    nethack_dir = "../nh/install/games/lib/nethackdir/"
elif platform == "darwin":
    nethack_path = "nethack/install/nethack" # relative path to nethack
    nethack_dir = "nethack/game"
elif platform == "win32":
    nethack_path = "nethack\\binary\\nethack.exe"
    nethack_dir = "C:\\msys64\\home\\Jonathan\\nethackrl\\nethack\\binary"

def launch_nh(socket):
    socket.send("launch".encode())
    socket.recv()

def kill_nh(socket):
    try:
        socket.send("Q".encode())
    except:
        # force quit
        # src : https://stackoverflow.com/questions/2940858/kill-process-by-name
        if platform in ['linux', 'linux2', 'darwin']:
            p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
            out, err = p.communicate()
            for line in out.splitlines():
                if nethack_path in str(line):
                    pid = int(line.split(None, 1)[0])
                    os.kill(pid, signal.SIGKILL)

def send_msg(socket, msg):
    assert socket is not None
    if type(msg) is not list:
        msg = [msg]
    
    #verboseprint("Sending", msg)
    for i, a in enumerate(msg):
        assert a is not None
        socket.send(a.encode(), zmq.NOBLOCK)
        if i < len(msg) - 1: # discard states until last action sent
            message = rcv_msg(socket)
            if "***dir***" not in message:
                return message
    return None

def rcv_msg(socket):
    message = socket.recv()
    return message.decode("cp437" if os.name == "nt" else "ISO-8859-1")
