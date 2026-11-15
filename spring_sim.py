
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
from torch import nn
from pysr import PySRRegressor

# ==========================================
# 1. Physical parameters
# ==========================================
torch.manual_seed(42)
np.random.seed(42)

m = 10.0
k = 250.0
g = 9.81

dt = 0.02
t_eval_total = np.arange(1000) * dt
n_train = 500

# ==========================================
# 2. Define spring-mass equations
# ==========================================
def spring_equations(t, y):
    x, v = y
    dx_dt = v
    dv_dt = -(k / m) * x
    return [dx_dt, dv_dt]


solution_total = solve_ivp(
    spring_equations,
    (0, 20),
    [1.0, 0.0],
    t_eval=t_eval_total,
    rtol=1e-9,
    atol=1e-11
)

if not solution_total.success:
    raise RuntimeError(solution_total.message)

total_displacements = solution_total.y[0]
total_velocities = solution_total.y[1]

clean_states = np.column_stack(
    (total_displacements, total_velocities)
)

# ==========================================
# 3. Generate noisy training data (0-10 s)
# ==========================================
train_states = clean_states[:n_train].copy()

noise = np.random.normal(
    0, 0.9, size=n_train
)
train_states[:, 1] += noise

x_data = train_states[:-1]
y_data = train_states[1:]

x_tensor = torch.tensor(
    x_data, dtype=torch.float32
)
y_tensor = torch.tensor(
    y_data, dtype=torch.float32
)

# ==========================================
# 4. Define neural network
# ==========================================
class SpringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2, 128)
        self.layer_2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_2(
            self.relu(self.layer_1(x))
        )


model = SpringModel()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3
)

epochs = 1000

for epoch in range(epochs):
    model.train()

    predictions = model(x_tensor)
    loss = loss_fn(predictions, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(
            f"Epoch: {epoch + 1} | "
            f"Loss: {loss.item():.6f}"
        )

# ==========================================
# 5. Filtering and autoregressive prediction
# ==========================================
model.eval()
combined_trajectory = []

with torch.inference_mode():
    best_fit_0_10 = model(x_tensor).cpu().numpy()

    for step in best_fit_0_10:
        combined_trajectory.append(step.copy())

    current_state = torch.tensor(
        best_fit_0_10[-1],
        dtype=torch.float32
    )

    future_steps = (
        len(t_eval_total) - len(best_fit_0_10) - 1
    )

    for _ in range(future_steps):
        next_state = model(current_state)
        current_state = next_state.squeeze(0)

        combined_trajectory.append(
            current_state.cpu().numpy().copy()
        )

combined_trajectory = np.asarray(
    combined_trajectory
)

prediction_times = t_eval_total[1:]

assert len(combined_trajectory) == len(prediction_times)

# ==========================================
# 6. Plot results
# ==========================================
plt.figure(figsize=(12, 6))

plt.scatter(
    t_eval_total[:n_train],
    train_states[:, 1],
    color="blue",
    alpha=0.2,
    s=10,
    label="Noisy Training Data (0-10 s)"
)

plt.plot(
    t_eval_total,
    total_velocities,
    color="black",
    linewidth=2,
    label="True Clean Velocity"
)

plt.plot(
    prediction_times,
    combined_trajectory[:, 1],
    "r--",
    linewidth=2,
    label="NN Output (Filter -> Predict)"
)

plt.axvline(
    10.0,
    color="purple",
    linestyle=":",
    linewidth=2,
    label="Training End / Prediction Start"
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title(
    "Spring-Mass NN: Filtering and Autoregressive Prediction"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 7. PySR symbolic regression
# ==========================================
def run_lagrangian_symbolic_regression(
    combined_trajectory, t_eval_total
):
    print("\n--- Processing Spring Data for PySR ---")

    dt = float(t_eval_total[1] - t_eval_total[0])

    X_raw = combined_trajectory[:-1]
    y_raw = combined_trajectory[1:]

    displacement = X_raw[:, 0]
    velocity = X_raw[:, 1]

    velocity_squared = velocity**2

    # Features: displacement, velocity, velocity squared
    X_symbolic = np.column_stack(
        (displacement, velocity, velocity_squared)
    )

    acceleration = (
        y_raw[:, 1] - velocity
    ) / dt

    model_pysr = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        variable_names=["x", "v", "v_sq"],
        model_selection="best"
    )

    print("Running PySR evolutionary search...")
    model_pysr.fit(X_symbolic, acceleration)

    print("\n--- Discovered Spring Equations ---")
    print(model_pysr.equations_)

    return model_pysr


model_pysr = run_lagrangian_symbolic_regression(
    combined_trajectory,
    t_eval_total
)
