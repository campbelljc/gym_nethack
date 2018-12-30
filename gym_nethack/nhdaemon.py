import os, sys, subprocess

import zmq

from gym_nethack.conn import nethack_path, nethack_dir

'''
def spawn_daemon(proc_id):
    """Spawn a completely detached subprocess (i.e., a daemon).
    source: https://stackoverflow.com/questions/972362/spawning-process-from-python/972383#972383
    """
    path_to_executable = sys.executable
    args = [sys.executable, '-m', 'nhdaemon', str(proc_id)]
    # fork the first time (to make a non-session-leader child process)
    try:
        pid = os.fork()
    except OSError as e:
        raise RuntimeError("1st fork failed: %s [%d]" % (e.strerror, e.errno))
    if pid != 0:
        # parent (calling) process is all done
        return

    # detach from controlling terminal (to make child a session-leader)
    os.setsid()
    try:
        pid = os.fork()
    except OSError as e:
        raise RuntimeError("2nd fork failed: %s [%d]" % (e.strerror, e.errno))
        raise Exception("%s [%d]" % (e.strerror, e.errno))
    if pid != 0:
        # child process is all done
        os._exit(0)

    # grandchild process now non-session-leader, detached from parent
    # grandchild process must now close all open files
    try:
        maxfd = os.sysconf("SC_OPEN_MAX")
    except (AttributeError, ValueError):
        maxfd = 1024

    for fd in range(maxfd):
        try:
           os.close(fd)
        except OSError: # ERROR, fd wasn't open to begin with (ignored)
           pass

    # redirect stdin, stdout and stderr to /dev/null
    os.open(os.devnull, os.O_RDWR)  # standard input (0)
    os.dup2(0, 1)
    os.dup2(0, 2)

    # and finally let's execute the executable for the daemon!
    try:
      os.execv(path_to_executable, args)
    except Exception:
      # oops, we're cut off from the world, let's just give up
      os._exit(255)
'''
def nh_daemon():
    proc_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nh_port  = 5555 + proc_id
    gym_port = 5555 - proc_id - 1
    print("Daemon launched for NH port", nh_port, "; listening on port", gym_port)
    context = zmq.Context()
    daemon_socket = context.socket(zmq.REP)
    daemon_socket.bind("tcp://*:" + str(gym_port))
    while True:
        message = daemon_socket.recv().decode("cp437" if os.name == "nt" else "utf-8")
        if 'exit' in message:
            break
        if 'test' in message:
            daemon_socket.send("done".encode())
            continue
        
        assert 'launch' in message
        daemon_socket.send("done".encode())
        if sys.platform == "win32":
            os.system("cd " + nethack_dir + " & NetHack.exe -port " + str(nh_port) + "\"")
        else:
            subprocess.call(nethack_path + " -port " + str(nh_port) + " &", shell=True)
    print("Received:", message)

if __name__ == '__main__':
    nh_daemon()
