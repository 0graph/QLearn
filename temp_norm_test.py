import numpy as np

class NormalizationParams:
    def __init__(self):
        self.n = 0          # count of values seen so far
        self.mean = 0.0     # mean of values seen so far
        self.M2 = 0.0       # sum of squares of differences from mean

    def calc_M2(self, arr: np.ndarray):
        mean = arr.mean()
        return np.sum((arr - mean) ** 2)

    # update based on array
    # Chan's parallel algorithm (see source above)
    def update_from_array(self, arr: np.ndarray):
        flat = arr.flatten()
        comb_n = self.n + len(flat)
        comb_delta = flat.mean() - self.mean
        comb_delta2 = self.M2 + self.calc_M2(flat) + (comb_delta ** 2) * self.n * len(flat) / comb_n
        self.n = comb_n
        self.mean += comb_delta * len(flat) / comb_n
        self.M2 = comb_delta2

    # update mean and std
    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2 

    def get_mean(self):
        return self.mean

    def get_std(self):
        return (self.M2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0
    
    def __str__(self):
        return f"Mean: {self.mean}, Std: {self.get_std()}, N: {self.n}"

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr2 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
arr3 = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
arr4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
norm_params1 = NormalizationParams()
norm_params2 = NormalizationParams()

for i in range(10):
    norm_params1.update(arr[i])
    
norm_params2.update_from_array(arr)

print(norm_params1)
print(norm_params2)

for i in range(10):
    norm_params1.update(arr2[i])

norm_params2.update_from_array(arr2)

print(norm_params1)
print(norm_params2)

for i in range(10):
    norm_params1.update(arr3[i])

norm_params2.update_from_array(arr3)

print(norm_params1)
print(norm_params2)

for i in range(30):
    norm_params1.update(arr4[i])

norm_params2.update_from_array(arr4)

print(norm_params1)
print(norm_params2)


