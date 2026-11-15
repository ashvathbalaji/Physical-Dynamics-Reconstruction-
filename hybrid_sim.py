
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
L0 = 1.0

dt = 0.02
t_eval_total = np.arange(1000) * dt
n_train = 500

# ==========================================
# 2. Define hybrid equations
# ==========================================
def hybrid_equations(t, y):
    r, vr, theta, omega = y

    # Prevent division by zero if the spring
    # approaches zero length.
    r_safe = max(r, 0.05)

    dr_dt = vr

    # Radial acceleration
    dvr_dt = (
        r_safe * omega**2
        - (k / m) * (r_safe - L0)
        - g * np.cos(theta)
    )

    dtheta_dt = omega

    # Angular acceleration
    domega_dt = (
        -(g / r_safe) * np.sin(theta)
        - 2 * vr * omega / r_safe
    )

    return [dr_dt, dvr_dt, domega_dt * 0 + domega_dt,
        domega_dt]


solution_total = solve_ivp(
    hybrid_equations,
    (0, 20),
    [L0, 0.0, np.pi / 4, 0.0],
    t_eval=t_eval_total,
    rtol=1e-9,
    atol=1e-11,
    max_step=0.01
)

if not solution_total.success:
    raise RuntimeError(solution_total.message)

clean_states = solution_total.y.T

# State columns: [r, vr, theta, omega]

# ==========================================
# 3. Generate noisy training data
# ==========================================
train_states = clean_states[:n_train].copy()

# Add Gaussian noise to both velocity measurements
train_states[:, 1] += np.random.normal(
    0, 0.9, size=n_train
)
train_states[:, 3] += np.random.normal(
    0, 0.9, size=n_train
)

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
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(4, 128)
        self.layer_2 = nn.Linear(128, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_2(
            self.relu(self.layer_1(x))
        )


model = HybridModel()

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
# 6. Plot radial and angular velocities
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

axes[0].scatter(
    t_eval_total[:n_train],
    train_states[:, 1],
    color="blue",
    alpha=0.2,
    s=10,
    label="Noisy Training Data"
)
axes[0].plot(
    t_eval_total,
    clean_states[:, 1],
    color="black",
    linewidth=2,
    label="True Radial Velocity"
)
axes[0].plot(
    prediction_times,
    combined_trajectory[:, 1],
    "r--",
    linewidth=2,
    label="NN Output"
)
axes[0].axvline(10, color="purple", linestyle=":")
axes[0].set_ylabel("Radial Velocity (m/s)")
axes[0].set_title("Hybrid System: Radial Motion")
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(
    t_eval_total[:n_train],
    train_states[:, 3],
    color="blue",
    alpha=0.2,
    s=10,
    label="Noisy Training Data"
)
axes[1].plot(
    t_eval_total,
    clean_states[:, 3],
    color="black",
    linewidth=2,
    label="True Angular Velocity"
)
axes[1].plot(
    prediction_times,
    combined_trajectory[:, 3],
    "r--",
    linewidth=2,
    label="NN Output"
)
axes[1].axvline(10, color="purple", linestyle=":")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Angular Velocity (rad/s)")
axes[1].set_title("Hybrid System: Angular Motion")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 7. PySR symbolic regression
# ==========================================
def run_hybrid_symbolic_regression(
    trajectory, t_eval_total
):
    print("\n--- Processing Hybrid Data for PySR ---")

    dt = float(t_eval_total[1] - t_eval_total[0])

    X_raw = trajectory[:-1]
    y_raw = trajectory[1:]

    r = X_raw[:, 0]
    vr = X_raw[:, 1]
    theta = X_raw[:, 2]
    omega = X_raw[:, 3]

    # Estimate both accelerations from consecutive states
    radial_acceleration = (y_raw[:, 1] - vr) / dt
    angular_acceleration = (y_raw[:, 3] - omega) / dt

    # Engineered features help PySR search for
    # the coupled radial and angular equations.
    X_symbolic = np.column_stack([
        r,
        vr,
        theta,
        omega,
        np.cos(theta),
        np.sin(theta),
        omega**2,
        r * omega**2,
        vr * omega / np.maximum(r, 0.05),
        np.sin(theta) / np.maximum(r, 0.05)
    ])

    variables = [
        "r",
        "vr",
        "theta",
        "omega",
        "cos_theta",
        "sin_theta",
        "omega_sq",
        "r_omega_sq",
        "vr_omega_over_r",
        "sin_over_r"
    ]

    def fit_equation(target, target_name):
        pysr_model = PySRRegressor(
            niterations=100,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "cos"],
            variable_names=variables,
            model_selection="best"
        )

        print(f"\nSearching for {target_name}...")
        pysr_model.fit(X_symbolic, target)

        print(f"\n--- Discovered {target_name} Equations ---")
        print(pysr_model.equations_)

        return pysr_model

    radial_model = fit_equation(
        radial_acceleration,
        "Radial Acceleration"
    )

    angular_model = fit_equation(
        angular_acceleration,
        "Angular Acceleration"
    )

    return radial_model, angular_model


radial_pysr, angular_pysr = run_hybrid_symbolic_regression(
    combined_trajectory,
    t_eval_total
)
