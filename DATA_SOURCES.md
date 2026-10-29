# 数据来源 / Data sources

作者提供的数据来源如下。本仓库仅发布代码、配置示例和说明，不提供任何原始数据、下载记录、处理后数据、汇总结果、预测文件或模型权重。

| 数据来源 / Source | 官方查询入口 / Official search portal |
|---|---|
| 中国临床试验注册中心（ChiCTR） / Chinese Clinical Trial Registry | https://www.chictr.org.cn/searchproj.html |
| ClinicalTrials.gov | https://clinicaltrials.gov/search?viewType=Card |
| 药物临床试验登记与信息公示平台 / Drug Clinical Trial Registration and Information Disclosure Platform | https://www.chinadrugtrials.org.cn/clinicaltrials.searchlist.dhtml |
| 药品审评中心（CDE）信息公开 / Center for Drug Evaluation information disclosure | https://www.cde.org.cn/main/xxgk/listpage/b40868b5e21c038a6aa8b4319d21b07d |

使用者需自行依据各平台的访问规则及使用条件获取所需数据。列出来源不代表本仓库授予数据使用或再分发许可。

## 与代码输入的关系 / Relationship to the code input

现有数据处理代码读取本地整理后的注册受理记录 Excel（存放在 ZIP 中），不是上述四个平台的采集器，也没有实现四库之间的记录链接或合并。上述列表由作者提供，用于说明数据来源；仅凭公开查询入口不能验证论文所用快照的具体提取日期、逐字段来源、跨库匹配方式或历史可用性。不得将本代码包描述为已验证的四库整合管线。

The sources above were supplied by the authors. The current pipeline consumes a locally prepared registration-application workbook inside a ZIP archive. It does not scrape these portals or implement cross-registry linkage. The source list alone does not verify the study snapshot's extraction dates, field-level provenance, linkage procedure or historical availability. Prepare authorized inputs in the schema described in docs/DATA.md before running the pipeline. Exact reproduction requires the corresponding original snapshot, which is not distributed here.
