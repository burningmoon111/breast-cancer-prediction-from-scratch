/**
 * ============================================================
 *  乳腺癌数据 - 多元线性回归预测模型 (C语言实现)
 *  数据集: Wisconsin Diagnostic Breast Cancer (WDBC)
 *  算法:   梯度下降法 多元线性回归
 *  适配:   JupyterLab 在线编程环境 (C kernel / 终端编译运行)
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* ==================== 常量定义 ==================== */
#define MAX_SAMPLES   600     /* 最大样本数 */
#define N_FEATURES    30      /* 特征数量 */
#define MAX_LINE_LEN  8192    /* 单行最大长度 */
#define MAX_ITER      5000    /* 最大迭代次数 */

/* ==================== 数据结构 ==================== */

/* 数据集结构 */
typedef struct {
    double X[MAX_SAMPLES][N_FEATURES];  /* 特征矩阵 */
    double y[MAX_SAMPLES];              /* 标签向量 (0=良性, 1=恶性) */
    int    n_samples;                    /* 样本数 */
    int    n_features;                   /* 特征数 */
} Dataset;

/* 归一化器 (保存均值和标准差用于训练集→测试集传递) */
typedef struct {
    double mean[N_FEATURES];
    double std[N_FEATURES];
} Normalizer;

/* 线性回归模型 */
typedef struct {
    double weights[N_FEATURES + 1];   /* w0(bias) + w1..w30 */
    double loss_history[MAX_ITER];    /* 损失历史记录 */
    int    n_iterations;              /* 实际迭代次数 */
    int    n_features;                /* 特征数 */
} LinearRegression;

/* ==================== 数据加载 ==================== */

/**
 * 加载CSV数据集
 * 文件格式: ID\tf1,f2,...,f30,label
 * 返回: 成功加载的样本数, 失败返回-1
 */
int load_csv(const char *filename, Dataset *data) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("[ERROR] 无法打开文件: %s\n", filename);
        printf("[提示] 请确保 breast_cancer.csv 在当前目录下\n");
        printf("[格式] 每行31个逗号分隔的数值: f1,f2,...,f30,label\n");
        return -1;
    }

    char line[MAX_LINE_LEN];
    int row = 0;

    while (fgets(line, sizeof(line), fp) && row < MAX_SAMPLES) {
        /* 去除换行符 */
        line[strcspn(line, "\r\n")] = '\0';

        /* 跳过空行 */
        if (strlen(line) < 10) continue;

        /* 直接解析逗号分隔的浮点数: feature1,feature2,...,feature30,label */
        char *token = strtok(line, ",");
        int col = 0;

        while (token && col < N_FEATURES + 1) {
            double val = atof(token);
            if (col < N_FEATURES) {
                data->X[row][col] = val;   /* 特征 */
            } else {
                data->y[row] = val;         /* 标签 (0=良性, 1=恶性) */
            }
            token = strtok(NULL, ",");
            col++;
        }

        /* 确保解析到足够的列 (30特征 + 1标签 = 31) */
        if (col >= N_FEATURES + 1) {
            row++;
        }
    }

    fclose(fp);
    data->n_samples  = row;
    data->n_features = N_FEATURES;

    printf("[INFO] 成功加载 %d 条样本, 每条 %d 个特征\n", row, N_FEATURES);

    /* 统计类别分布 */
    int benign = 0, malignant = 0;
    for (int i = 0; i < row; i++) {
        if (data->y[i] < 0.5) benign++;
        else                   malignant++;
    }
    printf("[INFO] 良性(B=0): %d (%.1f%%), 恶性(M=1): %d (%.1f%%)\n",
           benign, 100.0 * benign / row,
           malignant, 100.0 * malignant / row);

    return row;
}

/* ==================== 特征归一化 ==================== */

/**
 * 计算训练集的均值和标准差 (Z-score: x' = (x - mean) / std)
 */
void compute_normalizer(Dataset *data, Normalizer *norm) {
    for (int j = 0; j < data->n_features; j++) {
        /* 计算均值 */
        double sum = 0.0;
        for (int i = 0; i < data->n_samples; i++) {
            sum += data->X[i][j];
        }
        norm->mean[j] = sum / data->n_samples;

        /* 计算标准差 */
        double var_sum = 0.0;
        for (int i = 0; i < data->n_samples; i++) {
            double diff = data->X[i][j] - norm->mean[j];
            var_sum += diff * diff;
        }
        norm->std[j] = sqrt(var_sum / data->n_samples);

        /* 防止除零 */
        if (norm->std[j] < 1e-10) {
            norm->std[j] = 1.0;
        }
    }
}

/**
 * 使用已有的归一化器对数据集进行归一化
 */
void apply_normalization(Dataset *data, Normalizer *norm) {
    for (int i = 0; i < data->n_samples; i++) {
        for (int j = 0; j < data->n_features; j++) {
            data->X[i][j] = (data->X[i][j] - norm->mean[j]) / norm->std[j];
        }
    }
}

/* ==================== 训练/测试集划分 ==================== */

/**
 * 随机划分训练集和测试集
 * @param ratio 训练集比例 (如 0.8 表示80%训练)
 */
void train_test_split(Dataset *src, Dataset *train, Dataset *test, double ratio) {
    int n_train = (int)(src->n_samples * ratio);
    int n_test  = src->n_samples - n_train;

    train->n_samples  = n_train;
    train->n_features = src->n_features;
    test->n_samples   = n_test;
    test->n_features  = src->n_features;

    /* 创建随机索引 */
    int *indices = (int *)malloc(src->n_samples * sizeof(int));
    for (int i = 0; i < src->n_samples; i++) indices[i] = i;

    /* Fisher-Yates 洗牌 */
    srand(42);  /* 固定随机种子, 确保结果可复现 */
    for (int i = src->n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    /* 分配训练集 */
    for (int i = 0; i < n_train; i++) {
        int idx = indices[i];
        for (int j = 0; j < src->n_features; j++) {
            train->X[i][j] = src->X[idx][j];
        }
        train->y[i] = src->y[idx];
    }

    /* 分配测试集 */
    for (int i = 0; i < n_test; i++) {
        int idx = indices[i + n_train];
        for (int j = 0; j < src->n_features; j++) {
            test->X[i][j] = src->X[idx][j];
        }
        test->y[i] = src->y[idx];
    }

    free(indices);

    printf("[INFO] 训练集: %d 样本, 测试集: %d 样本 (划分比例: %.0f%%/%.0f%%)\n",
           n_train, n_test, ratio * 100, (1 - ratio) * 100);
}

/* ==================== 线性回归模型 ==================== */

/**
 * 初始化模型参数
 */
void model_init(LinearRegression *model, int n_features) {
    model->n_features   = n_features;
    model->n_iterations = 0;
    for (int i = 0; i <= n_features; i++) {
        model->weights[i] = 0.0;  /* 权重全部初始化为0 */
    }
    for (int i = 0; i < MAX_ITER; i++) {
        model->loss_history[i] = 0.0;
    }
}

/**
 * 单个样本预测: h(x) = w0 + w1*x1 + w2*x2 + ... + wn*xn
 */
double predict_one(LinearRegression *model, double *x) {
    double h = model->weights[0];  /* bias项 w0 */
    for (int j = 0; j < model->n_features; j++) {
        h += model->weights[j + 1] * x[j];
    }
    return h;
}

/**
 * 批量预测
 */
void predict_batch(LinearRegression *model, Dataset *data, double *predictions) {
    for (int i = 0; i < data->n_samples; i++) {
        predictions[i] = predict_one(model, data->X[i]);
    }
}

/**
 * 计算均方误差损失: MSE = (1/2m) * Σ(h(x_i) - y_i)²
 */
double compute_loss(LinearRegression *model, Dataset *data) {
    double total = 0.0;
    for (int i = 0; i < data->n_samples; i++) {
        double h = predict_one(model, data->X[i]);
        double error = h - data->y[i];
        total += error * error;
    }
    return total / (2.0 * data->n_samples);
}

/**
 * 梯度下降训练
 * @param lr       学习率
 * @param max_iter 最大迭代次数
 * @param tol      早停容忍度 (损失变化小于此值则停止)
 */
void gradient_descent(LinearRegression *model, Dataset *data,
                      double lr, int max_iter, double tol) {
    int m = data->n_samples;
    int n = data->n_features;

    printf("\n[训练] 开始梯度下降...\n");
    printf("        学习率=%.4f, 最大迭代=%d, 早停容忍度=%.6f\n", lr, max_iter, tol);

    double prev_loss = compute_loss(model, data);

    for (int iter = 0; iter < max_iter; iter++) {
        /* 计算梯度 */
        double gradient[N_FEATURES + 1] = {0.0};

        for (int i = 0; i < m; i++) {
            double h     = predict_one(model, data->X[i]);
            double error = h - data->y[i];

            gradient[0] += error;           /* bias的梯度 */
            for (int j = 0; j < n; j++) {
                gradient[j + 1] += error * data->X[i][j];
            }
        }

        /* 更新权重 */
        model->weights[0] -= (lr / m) * gradient[0];
        for (int j = 0; j < n; j++) {
            model->weights[j + 1] -= (lr / m) * gradient[j + 1];
        }

        /* 计算损失并检查收敛 */
        double current_loss = compute_loss(model, data);
        model->loss_history[iter] = current_loss;

        if (iter % 500 == 0) {
            printf("        [迭代 %4d] 损失 = %.6f\n", iter, current_loss);
        }

        /* 早停检查 */
        if (fabs(prev_loss - current_loss) < tol) {
            printf("        [迭代 %4d] 损失 = %.6f (收敛!)\n", iter, current_loss);
            model->n_iterations = iter + 1;
            break;
        }

        prev_loss = current_loss;
        model->n_iterations = iter + 1;
    }

    if (model->n_iterations >= max_iter) {
        printf("        [迭代 %4d] 达到最大迭代次数\n", max_iter - 1);
    }

    printf("[训练] 完成! 共迭代 %d 次, 最终损失 = %.6f\n",
           model->n_iterations, compute_loss(model, data));
}

/**
 * 训练模型 (完整流程: 归一化 + 梯度下降)
 */
void model_fit(LinearRegression *model, Dataset *train, double lr, int max_iter) {
    /* 1. 计算归一化参数并应用 */
    Normalizer norm;
    compute_normalizer(train, &norm);
    apply_normalization(train, &norm);

    /* 2. 梯度下降 */
    gradient_descent(model, train, lr, max_iter, 1e-7);

    /* 3. 将归一化器参数保存到weights的辅助位置 (用于后续测试集归一化)
     *    这里偷懒直接在训练函数里保留了norm变量返回...
     *    实际项目中应该把norm存到模型结构里。此处我们额外输出一个norm文件
     */
    FILE *fp = fopen("normalizer.txt", "w");
    for (int j = 0; j < train->n_features; j++) {
        fprintf(fp, "%.10f,%.10f\n", norm.mean[j], norm.std[j]);
    }
    fclose(fp);
}

/* ==================== 模型评估 ==================== */

/**
 * 计算模型评估指标
 * MSE:  均方误差
 * MAE:  平均绝对误差
 * R²:   决定系数
 * Acc:  准确率 (以0.5为阈值二分类)
 */
void model_evaluate(LinearRegression *model, Dataset *data,
                    double *mse, double *mae, double *r2,
                    double *accuracy, double *precision, double *recall) {
    int m        = data->n_samples;
    double sum_error  = 0.0;
    double sum_abs_err = 0.0;
    double y_mean = 0.0;

    /* 计算y均值 */
    for (int i = 0; i < m; i++) {
        y_mean += data->y[i];
    }
    y_mean /= m;

    double ss_res = 0.0;  /* 残差平方和 */
    double ss_tot = 0.0;  /* 总平方和 */

    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (int i = 0; i < m; i++) {
        double h     = predict_one(model, data->X[i]);
        double error = h - data->y[i];

        sum_error   += error * error;
        sum_abs_err += fabs(error);
        ss_res      += error * error;
        ss_tot      += (data->y[i] - y_mean) * (data->y[i] - y_mean);

        /* 二分类混淆矩阵 (阈值=0.5) */
        int pred_class = (h >= 0.5) ? 1 : 0;
        int true_class = (int)data->y[i];
        if (pred_class == 1 && true_class == 1) tp++;
        if (pred_class == 0 && true_class == 0) tn++;
        if (pred_class == 1 && true_class == 0) fp++;
        if (pred_class == 0 && true_class == 1) fn++;
    }

    *mse = sum_error / m;
    *mae = sum_abs_err / m;
    *r2  = 1.0 - (ss_res / (ss_tot + 1e-10));

    *accuracy  = (double)(tp + tn) / m;
    *precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
    *recall    = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
}

/* ==================== 可视化输出 ==================== */

/**
 * ASCII 终端可视化 - 损失曲线
 */
void plot_loss_ascii(LinearRegression *model) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════╗\n");
    printf("  ║           📉 损失函数收敛曲线 (ASCII)            ║\n");
    printf("  ╚══════════════════════════════════════════════════╝\n\n");

    int steps = model->n_iterations;
    if (steps > 80) steps = 80;  /* 终端宽度限制 */

    double max_loss = 0.0, min_loss = 1e10;
    for (int i = 0; i < model->n_iterations; i++) {
        if (model->loss_history[i] > max_loss) max_loss = model->loss_history[i];
        if (model->loss_history[i] < min_loss) min_loss = model->loss_history[i];
    }
    double range = max_loss - min_loss;
    if (range < 1e-10) range = 1.0;

    int step_size = model->n_iterations / steps;
    if (step_size < 1) step_size = 1;

    printf("  损失\n");
    for (int row = 20; row >= 0; row--) {
        double threshold = min_loss + range * row / 20.0;
        if (row == 20) printf("  %.4f │", max_loss);
        else if (row == 0) printf("  %.4f │", min_loss);
        else if (row == 10) printf("  %.4f │", (max_loss + min_loss) / 2);
        else printf("       │");

        for (int s = 0; s < steps; s++) {
            int iter_idx = s * step_size;
            if (iter_idx < model->n_iterations) {
                double loss = model->loss_history[iter_idx];
                if (loss <= threshold) printf("█");
                else printf(" ");
            }
        }
        printf("\n");
    }

    /* X轴 */
    printf("       └");
    for (int s = 0; s < steps; s++) printf("─");
    printf("\n       0");
    for (int s = 0; s < steps - 5; s++) printf(" ");
    printf("%d 迭代次数\n", model->n_iterations);
}

/**
 * ASCII 终端可视化 - 预测值 vs 真实值散点图
 */
void plot_predictions_ascii(double *y_true, double *y_pred, int n, const char *title) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════╗\n");
    printf("  ║  📊 %s  ║\n", title);
    printf("  ╚══════════════════════════════════════════════════╝\n\n");

    printf("  预测值\n");
    printf("  1.0 +");
    for (int i = 0; i < 50; i++) printf("-");
    printf("\n");

    int show_n = (n < 50) ? n : 50; /* 最多显示50个点 */
    int step = n / show_n;
    if (step < 1) step = 1;

    for (int row = 10; row >= 0; row--) {
        double level = row / 10.0;
        if (row == 10) printf("  %.1f │", 1.0);
        else if (row == 5) printf("  %.1f │", 0.5);
        else if (row == 0) printf("  %.1f │", 0.0);
        else printf("      │");

        for (int i = 0; i < show_n; i++) {
            int idx = i * step;
            if (idx < n) {
                double pred = y_pred[idx];
                if (fabs(pred - level) < 0.05) printf("*");
                else printf(" ");
            }
        }
        printf("\n");
    }

    printf("      +");
    for (int i = 0; i < show_n; i++) printf("-");
    printf("\n      0");
    for (int i = 0; i < show_n - 5; i++) printf(" ");
    printf("%d 样本序号\n", show_n);

    /* 统计阈值分类结果 */
    int correct = 0;
    for (int i = 0; i < n; i++) {
        int pred_class = (y_pred[i] >= 0.5) ? 1 : 0;
        int true_class = (int)y_true[i];
        if (pred_class == true_class) correct++;
    }
    printf("\n  正确分类: %d/%d (%.1f%%)\n", correct, n, 100.0 * correct / n);
}

/**
 * ASCII 终端可视化 - 特征权重重要性
 */
void plot_weights_ascii(LinearRegression *model) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════╗\n");
    printf("  ║           📊 特征权重重要性 (Top 15)             ║\n");
    printf("  ╚══════════════════════════════════════════════════╝\n\n");

    /* 特征名称(缩写) */
    char *feat_names[N_FEATURES] = {
        "mean_radius", "mean_texture", "mean_perim", "mean_area",
        "mean_smooth", "mean_compact", "mean_concav", "mean_cpts",
        "mean_symm", "mean_fdim",
        "se_radius", "se_texture", "se_perim", "se_area",
        "se_smooth", "se_compact", "se_concav", "se_cpts",
        "se_symm", "se_fdim",
        "worst_radius", "worst_texture", "worst_perim", "worst_area",
        "worst_smooth", "worst_compact", "worst_concav", "worst_cpts",
        "worst_symm", "worst_fdim"
    };

    /* 计算每个特征权重的绝对值 */
    double abs_w[N_FEATURES];
    double max_abs = 0.0;
    for (int j = 0; j < N_FEATURES; j++) {
        abs_w[j] = fabs(model->weights[j + 1]);
        if (abs_w[j] > max_abs) max_abs = abs_w[j];
    }

    /* 排序并输出Top 15 (简单冒泡排序索引) */
    int indices[N_FEATURES];
    for (int j = 0; j < N_FEATURES; j++) indices[j] = j;

    for (int i = 0; i < N_FEATURES - 1; i++) {
        for (int j = i + 1; j < N_FEATURES; j++) {
            if (abs_w[indices[j]] > abs_w[indices[i]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    printf("  %-3s %-20s %10s  %s\n", "排名", "特征名称", "权重值", "重要性条形图");
    printf("  %-3s %-20s %10s  %s\n", "---", "--------------------", "----------", "----------------");

    int top_n = (N_FEATURES < 15) ? N_FEATURES : 15;
    for (int i = 0; i < top_n; i++) {
        int idx = indices[i];
        double w = model->weights[idx + 1];
        double ratio = (max_abs > 0) ? abs_w[idx] / max_abs : 0.0;
        int bar_len = (int)(ratio * 30);

        printf("  %-3d %-20s %+10.4f  ", i + 1, feat_names[idx], w);
        for (int k = 0; k < bar_len; k++) printf("█");
        printf("\n");
    }
}

/* ==================== 保存预测结果 ==================== */

/**
 * 保存预测值vs真实值到CSV文件 (供Python matplotlib进一步可视化)
 */
void save_predictions_csv(double *y_true, double *y_pred, int n, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    fprintf(fp, "index,y_true,y_pred\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d,%.6f,%.6f\n", i, y_true[i], y_pred[i]);
    }
    fclose(fp);
    printf("[输出] 预测结果已保存至: %s\n", filename);
}

/**
 * 保存损失历史到CSV文件
 */
void save_loss_csv(LinearRegression *model, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) return;
    fprintf(fp, "iteration,loss\n");
    for (int i = 0; i < model->n_iterations; i++) {
        fprintf(fp, "%d,%.10f\n", i, model->loss_history[i]);
    }
    fclose(fp);
    printf("[输出] 损失历史已保存至: %s\n", filename);
}

/* ==================== 主程序 ==================== */

int main() {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     🔬 乳腺癌数据 - 多元线性回归预测模型 (C语言)        ║\n");
    printf("║     数据集: Wisconsin Diagnostic Breast Cancer           ║\n");
    printf("║     算法:   梯度下降法                                   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    /* ===== 1. 加载数据 ===== */
    printf("\n━━━ [步骤1] 加载数据集 ━━━\n");
    Dataset raw_data;
    int n = load_csv("breast_cancer.csv", &raw_data);
    if (n < 0) {
        printf("[ERROR] 数据加载失败, 程序退出。\n");
        printf("[提示] 请确保 breast_cancer.csv 在当前目录。\n");
        return 1;
    }

    /* ===== 2. 划分训练集/测试集 ===== */
    printf("\n━━━ [步骤2] 划分训练集/测试集 ━━━\n");
    Dataset train_data, test_data;
    train_test_split(&raw_data, &train_data, &test_data, 0.80);

    /* ===== 3. 初始化并训练模型 ===== */
    printf("\n━━━ [步骤3] 训练线性回归模型 ━━━\n");
    LinearRegression model;
    model_init(&model, N_FEATURES);
    model_fit(&model, &train_data, 0.01, MAX_ITER);

    /* 对测试集也进行归一化 (使用训练集的统计量) */
    Normalizer norm;
    FILE *nf = fopen("normalizer.txt", "r");
    if (nf) {
        for (int j = 0; j < N_FEATURES; j++) {
            fscanf(nf, "%lf,%lf", &norm.mean[j], &norm.std[j]);
        }
        fclose(nf);
        apply_normalization(&test_data, &norm);
    }

    /* ===== 4. 模型评估 ===== */
    printf("\n━━━ [步骤4] 模型评估 ━━━\n");

    /* 预测 */
    double *train_pred = (double *)malloc(train_data.n_samples * sizeof(double));
    double *test_pred  = (double *)malloc(test_data.n_samples  * sizeof(double));
    predict_batch(&model, &train_data, train_pred);
    predict_batch(&model, &test_data,  test_pred);

    /* 训练集评估 */
    double train_mse, train_mae, train_r2, train_acc, train_prec, train_rec;
    model_evaluate(&model, &train_data,
                   &train_mse, &train_mae, &train_r2,
                   &train_acc, &train_prec, &train_rec);

    /* 测试集评估 */
    double test_mse, test_mae, test_r2, test_acc, test_prec, test_rec;
    model_evaluate(&model, &test_data,
                   &test_mse, &test_mae, &test_r2,
                   &test_acc, &test_prec, &test_rec);

    /* 打印评估报告 */
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════╗\n");
    printf("  ║              📋 模型评估报告                      ║\n");
    printf("  ╠══════════════════════════════════════════════════╣\n");
    printf("  ║  指标            │  训练集   │  测试集           ║\n");
    printf("  ╠══════════════════╪═══════════╪═══════════════════╣\n");
    printf("  ║  MSE (均方误差)   │ %9.6f │ %9.6f        ║\n", train_mse, test_mse);
    printf("  ║  MAE (平均绝对误差)│ %9.6f │ %9.6f        ║\n", train_mae, test_mae);
    printf("  ║  R² (决定系数)    │ %9.4f │ %9.4f        ║\n", train_r2, test_r2);
    printf("  ║  准确率 Accuracy   │ %8.2f%% │ %8.2f%%         ║\n",
           train_acc * 100, test_acc * 100);
    printf("  ║  精确率 Precision  │ %8.2f%% │ %8.2f%%         ║\n",
           train_prec * 100, test_prec * 100);
    printf("  ║  召回率 Recall     │ %8.2f%% │ %8.2f%%         ║\n",
           train_rec * 100, test_rec * 100);
    printf("  ╚══════════════════╧═══════════╧═══════════════════╝\n");

    /* ===== 5. 可视化 (ASCII终端) ===== */
    printf("\n━━━ [步骤5] 可视化 (ASCII终端输出) ━━━\n");

    /* 损失曲线 */
    plot_loss_ascii(&model);

    /* 测试集预测vs真实 */
    plot_predictions_ascii(test_data.y, test_pred, test_data.n_samples,
                           "测试集: 预测值 vs 真实值");

    /* 特征权重 */
    plot_weights_ascii(&model);

    /* ===== 6. 保存结果文件 ===== */
    printf("\n━━━ [步骤6] 保存结果文件 ━━━\n");
    save_predictions_csv(test_data.y, test_pred, test_data.n_samples,
                         "predictions.csv");
    save_loss_csv(&model, "loss_history.csv");

    /* 清理 */
    free(train_pred);
    free(test_pred);

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║              ✅ 程序执行完毕!                            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    return 0;
}

