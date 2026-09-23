# 工商注册数据（H 盘，58 个省份分片，约 12.6 GB）→ 只保留与图谱主体/票据名单企业同名（含曾用名）的记录。
# 逐片 load，内存峰值为单片大小；名称规范化与 Python 端 norm_name 一致：NFKC、括号半角、去空白、去结尾“等”。
# 用法：先运行 31a_registry_targets.py 生成 targets.txt，再 Rscript --encoding=UTF-8 31b_registry_filter.R
suppressMessages({library(data.table); library(stringi)})
src_dir <- "H:/数据-工商注册企业_清洗后_2024-12"
out_dir <- "G:/ClaudeCode/graph-risk-model/cache/ext/registry"
targets <- readLines(file.path(out_dir, "targets.txt"), encoding = "UTF-8", warn = FALSE)
keep <- c("公司名称", "曾用名", "登记状态", "所属省份", "所属城市", "所属区县", "注册资本", "实缴资本", "成立日期",
          "核准日期", "企业规模", "参保人数", "统一社会信用代码", "国标行业门类", "公司类型")

norm <- function(x) {
  x <- stri_trans_nfkc(x)
  x <- chartr("（）【】", "()[]", x)
  x <- gsub("[[:space:]]+", "", x)
  x <- gsub("(等|等公司|等单位)$", "", x)
  x
}

files <- list.files(src_dir, pattern = "\\.Rdata$", full.names = TRUE)
files <- files[order(file.size(files))]
for (f in files) {
  o <- file.path(out_dir, sub("\\.Rdata$", ".csv", basename(f)))
  if (file.exists(o)) next
  t0 <- Sys.time()
  e <- new.env()
  nm <- tryCatch(load(f, envir = e), error = function(err) NULL)              # 个别分片在移动硬盘上读不出来：记下后跳过
  if (is.null(nm)) {
    cat(sprintf("%s  读取失败，跳过\n", basename(f)))
    write(basename(f), file.path(out_dir, "failed_shards.txt"), append = TRUE)
    rm(e); invisible(gc()); next
  }
  x <- as.data.table(get(nm[1], envir = e))
  rm(e); invisible(gc())
  x <- x[, intersect(keep, names(x)), with = FALSE]
  k1 <- norm(x[["公司名称"]])
  hit <- k1 %chin% targets
  if ("曾用名" %in% names(x)) {                                  # 曾用名：一次性拆开、统一规范化、批量匹配（不逐行循环）
    former <- x[["曾用名"]]
    idx <- which(!is.na(former) & nzchar(former))
    if (length(idx)) {
      parts <- stri_split_regex(former[idx], "[;；,，、|[:space:]]+")
      owner <- rep.int(seq_along(idx), lengths(parts))
      fh <- norm(unlist(parts, use.names = FALSE)) %chin% targets
      hit[idx[unique(owner[fh])]] <- TRUE
    }
  }
  y <- x[hit]
  y[, key := k1[hit]]
  fwrite(y, o)
  cat(sprintf("%s  %9d 行 → 命中 %6d  %.0fs\n", basename(f), nrow(x), nrow(y), as.numeric(Sys.time() - t0, units = "secs")))
  rm(x, y, k1, hit); invisible(gc())
}
cat("done\n")
