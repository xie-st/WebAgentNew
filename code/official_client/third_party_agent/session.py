"""Isolate each contestant task so a blocked algorithm cannot stall the client."""
import multiprocessing
import os
import signal
import time
from . import model


def _worker(pipe, factory, server, token, jid):
    if os.name != 'nt':
        os.setsid()
    agent = None
    try:
        while True:
            method, payload = pipe.recv()
            try:
                deadline = payload.get('execution_deadline') or time.time()+300
                with model.session(server, token, jid, payload.get('run_id'), deadline):
                    if method == 'reset':
                        agent = factory()
                        result = agent.reset(payload)
                    elif method == 'step':
                        result = agent.step(payload)
                    else:
                        if agent is not None and hasattr(agent, 'close'):
                            agent.close()
                        pipe.send((True, None))
                        return
                pipe.send((True, result))
            except Exception as exc:
                # Intentional: do not print exception text that may contain secrets.
                pipe.send((False, type(exc).__name__))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        pipe.close()


class AgentProcess:
    def __init__(self, factory, server, token, jid, cancelled=None):
        ctx = multiprocessing.get_context('spawn')
        parent, child = ctx.Pipe()
        self.pipe, self.run_id, self.cancelled = parent, None, cancelled
        self.process = ctx.Process(target=_worker,args=(child,factory,server,token,jid))
        self.process.start()
        child.close()

    def _call(self, method, request):
        deadline = min(request.get('execution_deadline') or time.time()+300, time.time()+300)
        if method == 'close':
            deadline = time.time()+1
        if time.time()>=deadline:
            self.kill()
            raise TimeoutError('Task execution deadline reached')
        try:
            self.pipe.send((method,request))
            while time.time()<deadline:
                if self.cancelled and self.cancelled():
                    self.kill()
                    raise RuntimeError('Task cancelled')
                if self.pipe.poll(min(.05,max(0,deadline-time.time()))):
                    ok, result = self.pipe.recv()
                    if not ok:
                        raise RuntimeError('Agent process error: '+str(result))
                    return result
                if not self.process.is_alive():
                    raise RuntimeError('Agent process exited unexpectedly')
        except (EOFError, BrokenPipeError, OSError):
            raise RuntimeError('Agent process connection closed') from None
        self.kill()
        raise TimeoutError('Task execution deadline reached')

    def reset(self, request):
        self.run_id=request['run_id']
        return self._call('reset',request)

    def step(self, request):
        return self._call('step',request)

    def kill(self):
        # A dead group leader can still leave a browser/algorithm child alive.
        if os.name != 'nt' and self.process.pid:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                if self.process.is_alive():
                    self.process.terminate()
        elif self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=.2)
        if os.name != 'nt' and self.process.pid:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif self.process.is_alive():
            self.process.kill()
        self.process.join(timeout=.2)

    def close(self):
        try:
            if self.process.is_alive():
                self._call('close',{'run_id':self.run_id})
        except (RuntimeError, TimeoutError):
            pass
        finally:
            self.kill()
            self.pipe.close()
