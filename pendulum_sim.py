
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

g = 9.81
L = 1.0

dt = 0.02
t_eval_total = np.arange(1000) * dt
n_train = 500

# ==========================================
# 2. Define pendulum equations
# ==========================================
def pendulum_equations(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]


solution_total = solve_ivp(
    pendulum_equations,
    (t_eval_total[0], t_eval_total[-1]),
    [np.pi / 4, 0.0],
    t_eval=t_eval_total,
    rtol=1e-9,
    atol=1e-11
)

if not solution_total.success:
    raise RuntimeError(solution_total.message)

total_angles = solution_total.y[0]
total_velocities = solution_total.y[1]

# State columns: [angle, angular velocity]
clean_states = np.column_stack(
    (total_angles, total_velocities)
)

# ==========================================
# 3. Generate noisy training data (0-10 s)
# ==========================================
train_states = clean_states[:n_train].copy()

# Add Gaussian noise to angular velocity only
noise = np.random.normal(
    0, 0.9, size=n_train
)
train_states[:, 1] += noise

# One-step inputs and targets
x_data = train_states[:-1]
y_data = train_states[1:]

x_tensor = torch.tensor(
    x_data, dtype=torch.float32
)
y_tensor = torch.tensor(
    y_data, dtype=torch.float32
)

# ==========================================
# 4. Neural network
# ==========================================
class PendulumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(4, 128)
        self.layer_2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        theta = x[:, 0:1]
        omega = x[:, 1:2]

        embedded_x = torch.cat(
            (
                theta,
                omega,
                torch.sin(theta),
                torch.cos(theta)
            ),
            dim=1
        )

        return self.layer_2(
            self.relu(self.layer_1(embedded_x))
        )


model = PendulumModel()

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

with torch.inference_mode():
    # One-step predictions for the training interval.
    best_fit_0_10 = model(x_tensor).cpu().numpy()

    # These predictions correspond to t = 0.02 ... 9.98 s.
    combined_trajectory = list(best_fit_0_10)

    # Start from the final one-step prediction,
    # corresponding to t = 9.98 s.
    current_state = torch.tensor(
        best_fit_0_10[-1],
        dtype=torch.float32
    )

    # Predict t = 10.00 ... 19.98 s.
    future_steps = len(t_eval_total) - n_train

    for _ in range(future_steps):
        next_state = model(current_state)
        current_state = next_state.squeeze(0)

        combined_trajectory.append(
            current_state.cpu().numpy().copy()
        )

combined_trajectory = np.asarray(
    combined_trajectory
)

# The predictions cover t = 0.02 ... 19.98 s.
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
    label="Noisy Training Data (0-10 s)",
    s=10
)

plt.plot(
    t_eval_total,
    total_velocities,
    color="black",
    alpha=0.7,
    linewidth=2,
    label="True Clean Velocity"
)

plt.plot(
    prediction_times,
    combined_trajectory[:, 1],
    color="red",
    linestyle="--",
    linewidth=2,
    label="NN Output (Filter -> Predict)"
)

plt.axvline(
    10.0,
    color="purple",
    linestyle=":",
    linewidth=2,
    label="Training End / Prediction Start (10 s)"
)

plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title(
    "Pendulum NN: Filtering and Autoregressive Prediction"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 7. Symbolic regression using PySR
# ==========================================
def run_lagrangian_symbolic_regression(
    combined_trajectory, t_eval_total
):
    print(
        "\n--- Processing Pendulum Trajectory for PySR ---"
    )

    dt = float(t_eval_total[1] - t_eval_total[0])

    # Consecutive predicted states
    X_raw = combined_trajectory[:-1]
    y_raw = combined_trajectory[1:]

    theta = X_raw[:, 0]
    omega = X_raw[:, 1]

    omega_squared = omega**2
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Engineered feature matrix
    X_symbolic = np.column_stack(
        (
            theta,
            omega,
            omega_squared,
            cos_theta,
            sin_theta
        )
    )

    # Estimate angular acceleration numerically
    acceleration = (
        y_raw[:, 1] - omega
    ) / dt

    model_pysr = PySRRegressor(
        niterations=100,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        variable_names=[
            "theta",
            "omega",
            "omega_sq",
            "cos_theta",
            "sin_theta"
        ],
        model_selection="best"
    )

    print("Running PySR evolutionary search...")
    model_pysr.fit(X_symbolic, acceleration)

    print("\n--- Discovered Equations ---")
    print(model_pysr.equations_)

    return model_pysr


model_pysr = run_lagrangian_symbolic_regression(
    combined_trajectory,
    t_eval_total
)
