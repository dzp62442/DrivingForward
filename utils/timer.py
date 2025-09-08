import time
import torch
from collections import OrderedDict
from functools import wraps

def timer_decorator(show=True, name=None):
    """
    装饰器用于统计函数运行时间，包含 CUDA 同步操作
    调用：
        @timer_decorator()
    Args:
        show (bool): 是否打印时间结果
        name (str): 自定义显示名称，默认使用函数名
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name if name is not None else func.__name__
            
            # 开始计时前同步CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 结束计时前同步CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            
            if show:
                print(f"[{func_name}] {elapsed_time:.4f} sec")
            
            return result
        return wrapper
    return decorator

class Timer:
    """多段计时器"""

    def __init__(self, newline=True):
        """
        Args:
            newline (bool): True to print on a new line, False to print on the same line.
        """
        self.newline = newline
        self.times = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        self.times.clear()

    def add(self, name='default', dt=None):
        if dt is not None:
            self.times[name] = dt

    def update(self, name='default'):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.time()
        dt = now - self.last_time
        self.times[name] = dt
        self.last_time = now

    def summary(self, text='Timer', show=True):
        total = 0.
        for name in self.times:
            total += self.times[name]

        if show:  # 打印到屏幕上
            print('[{}]'.format(text), end=' ')
            for name in self.times:
                dt = self.times[name]
                print('%s=%.4f' % (name, dt), end=' ')
            print('total=%.4f sec {%.2f FPS}' % (total, 1./total), end=' ')
            if self.newline:
                print(flush=True)
            else:
                print(end='\r', flush=True)

        times = self.times.copy()  # 返回一个副本
        self.reset()
        return total, times

# 示例用法
if __name__ == '__main__':
    for i in range(10):
        timer = Timer(newline=True)
        time.sleep(1)
        timer.update('one')
        time.sleep(1)
        timer.update('two')
        total, times = timer.summary(f'Demo{i}', show=True)
        print(times)