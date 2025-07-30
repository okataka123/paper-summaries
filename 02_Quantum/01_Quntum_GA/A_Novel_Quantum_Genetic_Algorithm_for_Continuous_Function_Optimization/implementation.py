import numpy as np
from typing import List, Tuple

# グレイコード変換関数：2進数をグレイコードへ変換
def binary_to_gray(binary: int) -> int:
    """2進数をグレイコードに変換する
    Args:
        binary: 2進表現の整数
    Returns:
        対応するグレイコードの整数
    """
    return binary ^ (binary >> 1)

# グレイコードを2進数へ変換
def gray_to_binary(gray: int) -> int:
    """グレイコードの整数を2進数に変換する
    Args:
        gray: グレイコード表現の整数
    Returns:
        対応する2進表現の整数
    """
    binary = gray
    shift = 1
    while gray >> shift:
        binary ^= gray >> shift
        shift += 1
    return binary

# グレイコード文字列を実数にデコードする
# 解空間の下限・上限にスケーリング
def decode_chromosome(
        chromosome: str, 
        lower_bound: float,
        upper_bound: float,
        n_bits: int,
        ) -> float:
    """グレイコードのビット列を実数値に変換する
    Args:
        chromosome: ビット列（文字列形式）
        lower_bound: 実数値の下限
        upper_bound: 実数値の上限
        n_bits: 変数あたりのビット数
    Returns:
        デコードされた実数値
    """
    value = gray_to_binary(int(chromosome, 2))
    max_val = 2 ** n_bits - 1
    return lower_bound + (upper_bound - lower_bound) * value / max_val

# 目的関数（Sphere関数）：最適値は x_i = 5 で最大 0
# 負の二乗和として表現し、最大化問題として扱う
def fitness_function(x: List[float]) -> float:
    """Sphere関数の適応度評価 (最大化問題として扱う)
    Args:
        x: 実数値の変数リスト
    Returns:
        負の2乗誤差(適応度)
    """
    # return -sum((xi - 5)**2 for xi in x)
    return 10-sum((xi - 5)**2 for xi in x)


# 量子個体クラス
class QuantumIndividual:
    """量子個体を表すクラス（角度ベクトルθで構成）"""

    def __init__(self, n_bits: int) -> None:
        """量子個体を初期化する（すべての角度をπ/4に設定)
        Args:
            n_bits: 量子ビットの総数
        """
        self.theta = np.full(n_bits, np.pi / 4)
        self.n_bits = n_bits

    # 観測により2進文字列を生成
    def measure(self) -> str:
        """量子ビットを観測して2進文字列を生成する
        Returns:
            観測結果の2進ビット列 (文字列形式)
        """
        probs = np.sin(self.theta) ** 2  # 各ビットが1になる確率
        return ''.join('1' if np.random.rand() < p else '0' for p in probs)

    # 回転操作（量子回転ゲート + Hεゲート）
    def rotate(
            self,
            best_bits: str,
            r: float,
            epsilon: float,
            ) -> None:
        """量子ビットを回転させる (Hεゲート適用含む)
        Args:
            best_bits: 最良解のビット列
            r: 回転角のスケーリング係数
            epsilon: 角度制限パラメータ (Hεゲート)
        """
        for i in range(self.n_bits):
            p = np.sin(self.theta[i]) ** 2  # 現在のビットの期待値
            delta = (int(best_bits[i]) - p) * 2 * r  # 期待値との差による回転量
            self.theta[i] += delta  # 角度を更新

            # Hεゲートにより角度が第1象限内に収まるよう制限
            if self.theta[i] < epsilon:
                self.theta[i] = epsilon
            elif self.theta[i] > (np.pi / 2 - epsilon):
                self.theta[i] = np.pi / 2 - epsilon

# NQGA本体クラス
class NQGA:
    """連続関数最適化のための新しい量子遺伝的アルゴリズム (NQGA)"""
    def __init__(
            self, 
            n_dims: int = 5, 
            n_bits: int = 10, 
            pop_size: int = 30, 
            max_iter: int = 500, 
            h: float = 0.01, 
            l: float = 0.01*np.pi, 
            epsilon: float = 0.01,
            ) -> None:
        """NQGAのパラメータを初期化する
        Args:
            n_dims: 変数の次元数
            n_bits: 各変数を表すビット数
            pop_size: 個体群のサイズ
            max_iter: 最大繰り返し回数
            h: 回転係数rの調整用定数
            l: 回転係数rの調整用定数
            epsilon: Hεゲートの下限パラメータ
        """
        self.n_dims: int = n_dims                  # 次元数（変数の数）
        self.n_bits: int = n_bits                  # 各変数あたりのビット数
        self.pop_size: int = pop_size              # 個体数
        self.max_iter: int = max_iter              # 最大世代数
        self.h: float = h                            # r調整用定数
        self.l: float = l                            # r調整用定数
        self.epsilon = epsilon                # Hεゲートしきい値
        self.lower_bound: float = 1                  # 探索空間の下限
        self.upper_bound: float = 10                 # 探索空間の上限
        self.population: List[QuantumIndividual] = [QuantumIndividual(n_bits * n_dims) for _ in range(pop_size)]  # 初期個体群生成

    # 最適化メインループ
    def optimize(self) -> Tuple[List[float], float]:
        """最適化処理を実行する

        Returns:
            最良解の実数リストとその関数値（目的関数の評価値）
        """
        best_solution = None
        best_fitness = -np.inf  # 初期最良値を負の無限大に

        for _ in range(self.max_iter):
            # 各個体を観測し、2進文字列（古典表現）を得る
            classical_pop = [ind.measure() for ind in self.population]
            decoded_pop = []  # 実数値ベクトル格納
            fitnesses = []    # 適応度格納

            for binary_str in classical_pop:
                # 各変数（次元）ごとにビットを切り出してデコード
                x = [decode_chromosome(binary_str[i*self.n_bits:(i+1)*self.n_bits],
                                       self.lower_bound, self.upper_bound, self.n_bits)
                     for i in range(self.n_dims)]
                fitness = fitness_function(x)
                decoded_pop.append((binary_str, x))
                fitnesses.append(fitness)

            # 最良個体のインデックス取得
            max_idx = np.argmax(fitnesses)

            # 最良個体を更新（収束判定付き）
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_solution = decoded_pop[max_idx][1]

                # for debug
                print('temporary best_fitness =', best_fitness)

            best_bits = decoded_pop[max_idx][0]  # 最良の2進文字列

            # 各個体を回転させる（個体ごとに異なるr）
            for k, ind in enumerate(self.population):
                r = k / self.pop_size + self.h * self.l
                ind.rotate(best_bits, r, self.epsilon)

        return best_solution, best_fitness  # 最良解とその関数値を返す

def main():
    nqga = NQGA()
    solution, value = nqga.optimize()
    print("最適解:", solution)
    print("関数値:", value)   


def test():
    pass

# 実行例
if __name__ == '__main__':
    main()
    # test()
