import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k_arms=10, epsilon=0.1, iterations=1000):
        self.k = k_arms
        self.epsilon = epsilon
        self.iterations = iterations
        
        # 1. Setup "Thế giới thực" (Casino)
        # Mỗi máy có một "tỉ lệ thắng ngầm" (true value) mà ta không biết
        # Ở đây ta giả lập nó bằng phân phối chuẩn
        self.q_true = np.random.randn(self.k) 
        
        # 2. Setup "Bộ não của Agent"
        # Q_est: Giá trị ước lượng ban đầu (thường là 0)
        self.q_est = np.zeros(self.k) 
        
        # N_count: Đếm số lần đã chọn mỗi máy (để tính 1/n)
        self.n_count = np.zeros(self.k) 
        
    def act(self):
        # Tung xúc xắc để quyết định Explore hay Exploit
        rand = np.random.rand()
        
        if rand < self.epsilon:
            # Exploration: Chọn ngẫu nhiên
            return np.random.randint(self.k)
        else:
            # Exploitation: Chọn máy có Q_est cao nhất hiện tại
            # (argmax trả về chỉ số của giá trị lớn nhất)
            return np.argmax(self.q_est)

    def step(self, action):
        # Mô phỏng việc kéo máy và nhận thưởng
        # Reward = True Value + Nhiễu (Noise)
        reward = np.random.normal(self.q_true[action], 1)
        return reward

    def update(self, action, reward):
        # --- ĐÂY LÀ PHẦN CỐT LÕI (Incremental Implementation) ---
        self.n_count[action] += 1
        step_size = 1.0 / self.n_count[action]
        
        # Form: New = Old + StepSize * (Target - Old)
        self.q_est[action] += step_size * (reward - self.q_est[action])
        
    def run(self):
        rewards = []
        for _ in range(self.iterations):
            action = self.act()            # 1. Chọn hành động
            reward = self.step(action)     # 2. Nhận thưởng
            self.update(action, reward)    # 3. Học (Cập nhật Q)
            rewards.append(reward)
        return rewards

# Chạy thử
bandit = Bandit(k_arms=10, epsilon=0.1, iterations=900000)
total_rewards = bandit.run()

print("True Values:", bandit.q_true)
print("Estimated Values:", bandit.q_est)
# Bạn sẽ thấy Estimated Values dần dần xấp xỉ True Values