import numpy as np
from typing import Callable, List, Any

class BenchmarkFunction:

    def __init__(
            self,
            upper: float = None,
            lower: float = None,
            dim: int = None,
            ) -> None:
        self.upper = upper
        self.lower = lower
        self.dim = dim

    def sphere(self, x: List[float]) -> float:
        """
        論文 A Novel Quantum Genetic Algorithm for Continuous Function Optimization 
        https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c74b1956cdb0800b2d0dac10c4ecd51a54985afd
        検証用関数 F_1 5次元の変数を想定
        定義域: 1 <= x_i <= 10
        備考: 
        """
        return 10 - sum((xi - 5)**2 for xi in x)

    def schwefel(self, x: List[float]) -> float:
        """
        論文 A Novel Quantum Genetic Algorithm for Continuous Function Optimization 
        https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c74b1956cdb0800b2d0dac10c4ecd51a54985afd
        検証用関数 F_2 5次元の変数を想定
        定義域: |x_i| <= 10
        備考: 
        """
        return 10 - sum(sum(x[j] for j in range(i))**2 for i in range(5))

    def rosenbrock(self, x: List[float]) -> float:
        """
        論文 A Novel Quantum Genetic Algorithm for Continuous Function Optimization 
        https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c74b1956cdb0800b2d0dac10c4ecd51a54985afd
        検証用関数 F_3 2次元の変数を想定
        定義域: |x_i| <= 5.12
        備考: 
        """
        ret = 10 - (100*(x[1]-x[0]**2)**2 + (x[0]-1)**2)
        return ret

    def shekel(self, x: List[float]) -> float:
        """
        論文 A Novel Quantum Genetic Algorithm for Continuous Function Optimization 
        https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c74b1956cdb0800b2d0dac10c4ecd51a54985afd
        検証用関数 F_4
        定義域: |x_i| <= 65.536
        備考: 

        参考: https://al-roomi.org/benchmarks/unconstrained/2-dimensions/7-shekel-s-foxholes-function
        """
        a = np.array([
            [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
            [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]
        ])
        return 1/500 + sum([1.0 / (j + (sum((x[i] - a[i][j])**6 for i in range(2)))) for j in range(25)])


    def sinc(self, x: List[float]) -> float: 
        """
        論文 A Novel Quantum Genetic Algorithm for Continuous Function Optimization 
        https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c74b1956cdb0800b2d0dac10c4ecd51a54985afd
        検証用関数 F_5
        備考: 

        定義域: 1 <= x_i <= 10
        """
        numerator = np.sin(sum(abs(x[i]-5)) for i in range(5))
        denominator = sum(abs(x[i]-5) for i in range(5))
        return numerator/denominator


    def sphere2(self):
        """
        """
        pass



def visualize(func: Callable[..., Any]) -> Any:
    pass


def main():
    bf = BenchmarkFunction()
    ret = bf.sphere(np.array([1, 2, 3, 4, 5]))
    print(ret)


if __name__ == "__main__":
    main()