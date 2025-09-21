import matplotlib.pyplot as plt

def plot_phase_portrait(true_states, pred_states, title="Phase Portrait"):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(true_states[:,0], true_states[:,1], true_states[:,2], lw=1, label="Ground Truth", alpha=0.7)
    ax.plot(pred_states[:,0], pred_states[:,1], pred_states[:,2], lw=1, label="Model Prediction", alpha=0.7)
    ax.set_title(title)
    ax.legend()
    plt.show()

def rollout_model(model, init_seq, steps=200):
    preds = []
    seq = init_seq.clone().unsqueeze(0)  # (1, seq_len, 3)
    for _ in range(steps):
        with torch.no_grad():
            pred = model(seq).cpu().numpy()
        preds.append(pred[0])
        seq = torch.cat([seq[:,1:,:], torch.tensor(pred).unsqueeze(0)], dim=1)
    return np.array(preds)

# 拿 test_data 前 seq_len 步當 seed
init_seq = torch.tensor(test_data[:10], dtype=torch.float32)
preds = rollout_model(naive_module, init_seq)

plot_phase_portrait(test_data[:200], preds, title="Naive vs Ground Truth")
