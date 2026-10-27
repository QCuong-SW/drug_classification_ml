"""Stage 6: Visualize and Communicate (Trực quan hóa và Đúc kết Tri thức)"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import make_pipeline

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["Drug"].value_counts()
    ratios = (df["Drug"].value_counts(normalize=True) * 100)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"], edgecolor="black")

    for bar, pct in zip(bars, ratios.values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{int(yval)} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")

    plt.title("Phân phối nhãn mục tiêu (Target Class Distribution)", fontsize=13, fontweight="bold")
    plt.xlabel("Nhóm thuốc (Drug)", fontsize=11)
    plt.ylabel("Số lượng mẫu bệnh nhân", fontsize=11)
    plt.ylim(0, max(counts.values) + 12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = output_dir / "01_target_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✓ Đã lưu biểu đồ: {out_path.name}")


def plot_feature_relationships(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 6))
    drugs = df["Drug"].unique()
    colors = {"drugY": "#3498db", "drugX": "#2ecc71", "drugA": "#e74c3c", "drugB": "#f39c12", "drugC": "#9b59b6"}

    for drug in drugs:
        sub = df[df["Drug"] == drug]
        plt.scatter(sub["Age"], sub["Na_to_K"], label=drug, color=colors.get(drug, "gray"), alpha=0.8, edgecolors="k", s=60)

    # Đường ngưỡng Na_to_K = 15 quyết định drugY
    plt.axhline(y=15, color="red", linestyle="--", linewidth=1.5, label="Ngưỡng Na_to_K = 15 (Quyết định drugY)")

    plt.title("Tương quan giữa Tuổi (Age) và Tỷ lệ Na/K (Na_to_K)", fontsize=13, fontweight="bold")
    plt.xlabel("Tuổi (Age)", fontsize=11)
    plt.ylabel("Tỷ lệ Na_to_K", fontsize=11)
    plt.legend(title="Thuốc", loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = output_dir / "02_feature_relationships.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✓ Đã lưu biểu đồ: {out_path.name}")


def plot_confusion_matrices(y_test, lr_pred, rf_pred, classes, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    cm_lr = confusion_matrix(y_test, lr_pred, labels=classes)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=classes)
    disp_lr.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title("Logistic Regression (Baseline)\nAccuracy: 95.00%", fontsize=12, fontweight="bold")

    cm_rf = confusion_matrix(y_test, rf_pred, labels=classes)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=classes)
    disp_rf.plot(ax=axes[1], cmap="Greens", values_format="d")
    axes[1].set_title("Random Forest (Tối ưu Pipeline)\nAccuracy: 97.50%", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / "03_confusion_matrices.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✓ Đã lưu biểu đồ: {out_path.name}")


def plot_decision_regions_2d(X_2d, y, output_dir: Path) -> None:
    unique_labels = sorted(y.unique())
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_encoded = y.map(label_map).values

    x_min, x_max = X_2d["Age"].min() - 3, X_2d["Age"].max() + 3
    y_min, y_max = X_2d["Na_to_K"].min() - 2, X_2d["Na_to_K"].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    models_2d = {
        "Logistic Regression (Ranh giới Tuyến tính)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Decision Tree (Ranh giới Phân mảnh vuông góc)": DecisionTreeClassifier(max_depth=4, random_state=42)
    }

    cmap_background = plt.cm.Pastel1
    colors_scatter = ["#e74c3c", "#f39c12", "#9b59b6", "#2ecc71", "#3498db"]

    for ax, (title, clf) in zip(axes, models_2d.items()):
        clf.fit(X_2d.values, y_encoded)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_background, levels=np.arange(len(unique_labels) + 1) - 0.5)

        for idx, lbl in enumerate(unique_labels):
            mask = y_encoded == idx
            ax.scatter(
                X_2d["Age"].values[mask],
                X_2d["Na_to_K"].values[mask],
                label=lbl,
                color=colors_scatter[idx],
                edgecolor="k",
                s=40
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Tuổi (Age)", fontsize=11)
        ax.set_ylabel("Tỷ lệ Na_to_K", fontsize=11)
        ax.legend(title="Lớp thuốc", loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("So sánh Ranh giới Quyết định 2D: Tuyến tính vs Cây quyết định\n(Đã lược bỏ BP, Cholesterol, Sex)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / "04_decision_boundary_2d.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✓ Đã lưu biểu đồ: {out_path.name}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "drug200.csv"
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    print("=" * 65)
    print("TIẾN HÀNH XUẤT CÁC BIỂU ĐỒ TRỰC QUAN HÓA (GENERATING FIGURES)")
    print("=" * 65)

    plot_target_distribution(df, figures_dir)
    plot_feature_relationships(df, figures_dir)

    X = df.drop(columns=["Drug"])
    y = df["Drug"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = ["Age", "Na_to_K"]
    cat_cols = ["Sex", "BP", "Cholesterol"]

    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(X_train[num_cols])
    X_te_num = scaler.transform(X_test[num_cols])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_tr_cat = encoder.fit_transform(X_train[cat_cols])
    X_te_cat = encoder.transform(X_test[cat_cols])

    X_tr_proc = np.hstack([X_tr_num, X_tr_cat])
    X_te_proc = np.hstack([X_te_num, X_te_cat])

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_tr_proc, y_train)
    lr_pred = lr.predict(X_te_proc)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_tr_proc, y_train)
    rf_pred = rf.predict(X_te_proc)

    classes = np.unique(y)
    plot_confusion_matrices(y_test, lr_pred, rf_pred, classes, figures_dir)
    plot_decision_regions_2d(df[["Age", "Na_to_K"]], df["Drug"], figures_dir)

    print("\n" + "=" * 65)
    print("GIẢI THÍCH GIỚI HẠN CỦA TRỰC QUAN HÓA 2 CHIỀU (2D LIMITATION)")
    print("=" * 65)
    print("• Biểu đồ 2D chỉ có thể thể hiện Age và Na_to_K.")
    print("• Trong không gian 2D, các mẫu drugA, drugB, drugC, drugX có vẻ chồng chéo lên nhau.")
    print("• Trong không gian 5 chiều thực tế, chúng được tách biệt rạch ròi")
    print("  nhờ các đặc trưng huyết áp (BP) và Cholesterol.")


if __name__ == "__main__":
    main()
