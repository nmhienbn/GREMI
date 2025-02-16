library(WGCNA)
library(data.table)
library(ggplot2)
library(dplyr)

# Load dữ liệu
data_tr <- read.csv("luad_train.csv")
data_te <- read.csv("luad_test.csv")

# Bỏ cột STT (cột 1) và cột Label (cột cuối)
tr_omic <- data_tr[, 2:(ncol(data_tr)-1)]  # Giữ lại chỉ các cột chứa dữ liệu omic
tr_labels <- data_tr[, ncol(data_tr)]  # Lấy cột cuối cùng làm nhãn
te_omic <- data_te[, 2:(ncol(data_te)-1)]  # Giữ lại chỉ các cột chứa dữ liệu omic
te_labels <- data_te[, ncol(data_te)]  # Lấy cột cuối cùng làm nhãn

# Chọn power tối ưu cho dữ liệu
powers <- c(1:20)
sft <- pickSoftThreshold(data_tr, powerVector = powers, verbose = 5)
softPower <- sft$powerEstimate  # Chọn power tối ưu

# Tạo exp_adj1: Ma trận adjacency
exp_adj1 <- adjacency(data_tr, power = softPower)

# Tạo exp_adj2: Ma trận TOM
TOM <- TOMsimilarity(exp_adj1)
exp_adj2 <- TOM

# Tạo exp_adj3: Ma trận dissimilarity (1 - TOM)
exp_adj3 <- 1 - TOM

