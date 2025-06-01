import numpy as np

# グレイコード変換関数：2進数をグレイコードへ変換
def binary_to_gray(binary):
    return binary ^ (binary >> 1)

# グレイコードを2進数へ変換
def gray_to_binary(gray):
    binary = gray
    shift = 1
    while gray >> shift:
        binary ^= gray >> shift
        shift += 1
    return binary

# グレイコード文字列を実数にデコードする
# 解空間の下限・上限にスケーリング
def decode_chromosome(chromosome, lower_bound, upper_bound, n_bits):
    value = gray_to_binary(int(chromosome, 2))
    max_val = 2 ** n_bits - 1
    return lower_bound + (upper_bound - lower_bound) * value / max_val

# 目的関数（Sphere関数）：最適値は x_i = 5 で最大 0
# 負の二乗和として表現し、最大化問題として扱う
def fitness_function(x):
    # return -sum((xi - 5)**2 for xi in x)
    return 10-sum((xi - 5)**2 for xi in x)


# 量子個体クラス
class QuantumIndividual:
    def __init__(self, n_bits):
        # 各量子ビットの角度θをπ/4に初期化（等確率）
        self.theta = np.full(n_bits, np.pi / 4)
        self.n_bits = n_bits

    # 観測により2進文字列を生成
    def measure(self):
        probs = np.sin(self.theta) ** 2  # 各ビットが1になる確率
        return ''.join('1' if np.random.rand() < p else '0' for p in probs)

    # 回転操作（量子回転ゲート + Hεゲート）
    def rotate(self, best_bits, r, epsilon):
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
    def __init__(self, n_dims=5, n_bits=10, pop_size=30, max_iter=500, h=0.01, l=0.01*np.pi, epsilon=0.01):
        self.n_dims = n_dims                  # 次元数（変数の数）
        self.n_bits = n_bits                  # 各変数あたりのビット数
        self.pop_size = pop_size              # 個体数
        self.max_iter = max_iter              # 最大世代数
        self.h = h                            # r調整用定数
        self.l = l                            # r調整用定数
        self.epsilon = epsilon                # Hεゲートしきい値
        self.lower_bound = 1                  # 探索空間の下限
        self.upper_bound = 10                 # 探索空間の上限
        self.population = [QuantumIndividual(n_bits * n_dims) for _ in range(pop_size)]  # 初期個体群生成

    # 最適化メインループ
    def optimize(self):
        best_solution = None
        best_fitness = -np.inf  # 初期最良値を負の無限大に

        for t in range(self.max_iter):
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
                if -best_fitness < 1e-2:
                    break  # 十分最適に近づいたら終了

            best_bits = decoded_pop[max_idx][0]  # 最良の2進文字列

            # 各個体を回転させる（個体ごとに異なるr）
            for k, ind in enumerate(self.population):
                r = k / self.pop_size + self.h * self.l
                ind.rotate(best_bits, r, self.epsilon)

        return best_solution, -best_fitness  # 最良解とその関数値を返す

def main():
    nqga = NQGA()
    solution, value = nqga.optimize()
    print("最適解:", solution)
    print("関数値:", value)   


def test():
    print(binary_to_gray(6))
    


# 実行例
if __name__ == '__main__':
    # main()
    test()
