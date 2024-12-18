import numpy as np


class AliasTable:
    """Walker's Alias Method"""

    def __init__(self):
        self.p = np.empty(1)
        self.a = np.empty(1)
        self.n = 0

    def build(self, wgt):
        if not isinstance(wgt, np.ndarray):
            wgt = np.array(wgt)
        self.n: int = len(wgt)  # class num
        self.p = (wgt / np.mean(wgt)).tolist()  # 平均で割る
        self.a, hl = [-1] * self.n, [0] * self.n
        left, right = 0, self.n - 1
        for i in range(self.n):
            if self.p[i] < 1:  # 平均より小さい
                hl[left] = i
                left += 1
            else:  # 平均より大きいグループ
                hl[right] = i
                right -= 1
        # Run Robin-hood algorithm; steal from the rich and fill poor pockets
        while left > 0 and right < self.n - 1:
            small, big = hl[left - 1], hl[right + 1]  # l:平均より小さい方, r:平均より大きい方
            self.a[small] = big
            self.p[big] -= 1 - self.p[small]  # lが合計1になるように大きい方を分ける
            if self.p[big] < 1:  # 足りなくなったら
                hl[left - 1] = big  # 小さいグループへ移動
                right += 1
            else:  # 大きいほうが残っていれば
                left -= 1  # 次の小さい方へ分ける

    def sample(self, nn=1):
        rr = np.random.rand(nn) * self.n
        ii = np.int32(np.floor(rr))
        rr -= ii
        return np.array([i if r < self.p[i] else self.a[i] for i, r in zip(ii, rr, strict=False)], dtype=int)
