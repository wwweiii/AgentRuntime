from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelResponse:
    content: str
    latency_s: float
    prompt_tokens_est: int
    output_tokens_est: int
    raw: dict[str, Any] | None = None


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3.5:4b", mock: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mock = mock

    def chat(self, system_prompt: str, user_prompt: str, model: str | None = None, timeout_s: int = 120) -> ModelResponse:
        start = time.time()
        if self.mock:
            content = self._mock_response(system_prompt, user_prompt)
            return ModelResponse(
                content=content,
                latency_s=time.time() - start,
                prompt_tokens_est=max(1, (len(system_prompt) + len(user_prompt)) // 3),
                output_tokens_est=max(1, len(content) // 3),
                raw={"mock": True},
            )

        body = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        content = raw.get("message", {}).get("content", "")
        return ModelResponse(
            content=content,
            latency_s=time.time() - start,
            prompt_tokens_est=max(1, (len(system_prompt) + len(user_prompt)) // 3),
            output_tokens_est=max(1, len(content) // 3),
            raw=raw,
        )

    # ── extended mock keyword table ──────────────────────────────────────
    _MOCK_TABLE: dict[str, str] = {
        "planner": "计划：1.需求分析 2.架构设计 3.模块拆分 4.接口定义 5.实现 6.集成测试 7.交付评审。依赖：architect,developer,tester。风险：需求变更、资源不足、集成冲突。",
        "规划": "计划：1.需求分析 2.架构设计 3.模块拆分 4.接口定义 5.实现 6.集成测试 7.交付评审。依赖：architect,developer,tester。风险：需求变更、资源不足、集成冲突。",
        "requirement": "需求分析结果：功能需求F1-F5已明确，非功能需求NFR1-NFR3已文档化。边界条件已在附录列出。建议架构师评审后进行技术方案设计。",
        "需求": "需求分析结果：功能需求F1-F5已明确，非功能需求NFR1-NFR3已文档化。边界条件已在附录列出。建议架构师评审后进行技术方案设计。",
        "architect": "架构设计：采用分层架构，表示层-业务层-持久层分离。模块间通过定义良好的接口通信，关键路径已标注性能预算。",
        "架构": "架构设计：采用分层架构，表示层-业务层-持久层分离。模块间通过定义良好的接口通信，关键路径已标注性能预算。",
        "frontend": "前端方案：React+TypeScript组件树，状态管理使用Context+Reducer模式。关键页面骨架已定义，API契约对齐后端Swagger文档。",
        "前端": "前端方案：React+TypeScript组件树，状态管理使用Context+Reducer模式。关键页面骨架已定义，API契约对齐后端Swagger文档。",
        "backend": "后端方案：FastAPI路由分层，Service-Repository模式，数据库连接池配置已优化。API版本策略：/api/v1/统一前缀。",
        "后端": "后端方案：FastAPI路由分层，Service-Repository模式，数据库连接池配置已优化。API版本策略：/api/v1/统一前缀。",
        "database": "数据库设计：ER模型共8个核心实体，索引策略覆盖高频查询路径。读写分离与分库分表方案已评估。",
        "数据库": "数据库设计：ER模型共8个核心实体，索引策略覆盖高频查询路径。读写分离与分库分表方案已评估。",
        "security": "安全审计报告：发现3个中危项（CORS配置、Token刷新策略、日志脱敏），1个低危项（错误信息泄露）。建议全部在Sprint内修复。",
        "安全": "安全审计报告：发现3个中危项（CORS配置、Token刷新策略、日志脱敏），1个低危项（错误信息泄露）。建议全部在Sprint内修复。",
        "performance": "性能评审：P99延迟在目标范围内，但峰值QPS下数据库连接池有瓶颈。建议增加连接池上限并引入Redis缓存热点数据。",
        "性能": "性能评审：P99延迟在目标范围内，但峰值QPS下数据库连接池有瓶颈。建议增加连接池上限并引入Redis缓存热点数据。",
        "tester": "测试结果：单元测试覆盖率82%，集成测试通过率95%。发现一个可修复问题：边界条件缺少空数组输入验证。RUNTIME_DYNAMIC_TASK:debugger",
        "测试": "测试结果：单元测试覆盖率82%，集成测试通过率95%。发现一个可修复问题：边界条件缺少空数组输入验证。RUNTIME_DYNAMIC_TASK:debugger",
        "debugger": "修复方案：已定位根因为输入校验层未处理零值边界。修复方式：在validate_input()中增加guard子句。状态：已修复，回归测试通过。",
        "修复": "修复方案：已定位根因为输入校验层未处理零值边界。修复方式：在validate_input()中增加guard子句。状态：已修复，回归测试通过。",
        "reviewer": "评审结论：整体架构清晰，模块职责分离良好。建议补充：(1)关键路径性能指标基线 (2)错误处理策略文档 (3)上下文复用率监控。",
        "评审": "评审结论：整体架构清晰，模块职责分离良好。建议补充：(1)关键路径性能指标基线 (2)错误处理策略文档 (3)上下文复用率监控。",
        "coder": "代码实现：已完成CLI入口、任务模型定义、上下文存储和测试框架搭建。模块间通过依赖注入解耦，关键路径已添加类型标注。",
        "代码": "代码实现：已完成CLI入口、任务模型定义、上下文存储和测试框架搭建。模块间通过依赖注入解耦，关键路径已添加类型标注。",
        "summarizer": "汇总报告：\n## 执行概览\n- 所有阶段均已完成，产出物已就绪\n## 关键决策\n- 技术栈选型已确认\n- 架构方案已通过评审\n## 遗留风险\n- 峰值性能需进一步压测验证\n## 下一步\n- 进入Beta发布流程",
        "汇总": "汇总报告：\n## 执行概览\n- 所有阶段均已完成，产出物已就绪\n## 关键决策\n- 技术栈选型已确认\n- 架构方案已通过评审\n## 遗留风险\n- 峰值性能需进一步压测验证\n## 下一步\n- 进入Beta发布流程",
        "translator": "翻译结果：源文档共解析出12个翻译单元。术语表已应用于所有专有名词。翻译记忆库命中率约34%。输出格式与源文档段落结构一致。",
        "翻译": "翻译结果：源文档共解析出12个翻译单元。术语表已应用于所有专有名词。翻译记忆库命中率约34%。输出格式与源文档段落结构一致。",
        "cultural": "文化适配审查：3处习语已替换为本地化表达，2处视觉隐喻已调整。品牌名称在所有目标语言中保留原名，法律条款已咨询当地合规。",
        "文化": "文化适配审查：3处习语已替换为本地化表达，2处视觉隐喻已调整。品牌名称在所有目标语言中保留原名，法律条款已咨询当地合规。",
        "formatter": "格式化完成：所有目标语言文档已统一字体、段落间距、页眉页脚。PDF和HTML双格式输出已生成，目录超链接已验证。",
        "格式": "格式化完成：所有目标语言文档已统一字体、段落间距、页眉页脚。PDF和HTML双格式输出已生成，目录超链接已验证。",
        "publisher": "发布完成：文档已推送至CDN节点，版本号v2.3.0。各语言入口URL已验证可访问，SEO元数据已注入。发布通知已发送至订阅列表。",
        "发布": "发布完成：文档已推送至CDN节点，版本号v2.3.0。各语言入口URL已验证可访问，SEO元数据已注入。发布通知已发送至订阅列表。",
        "business": "业务分析：目标市场TAM预估500M USD。核心用户画像3类，关键业务流程4条。MVP范围已划定，建议分3个Phase迭代交付。",
        "业务": "业务分析：目标市场TAM预估500M USD。核心用户画像3类，关键业务流程4条。MVP范围已划定，建议分3个Phase迭代交付。",
        "api": "API设计：RESTful资源建模完成，共12个资源端点。请求/响应Schema已定义，错误码采用RFC 7807 Problem Details规范。分页、排序、过滤均通过查询参数实现。",
        "payment": "支付系统设计：支持多渠道（微信、支付宝、银联、Visa/MC），采用策略模式封装支付渠道差异。幂等键保障重复提交安全，对账流程每6小时运行。",
        "支付": "支付系统设计：支持多渠道（微信、支付宝、银联、Visa/MC），采用策略模式封装支付渠道差异。幂等键保障重复提交安全，对账流程每6小时运行。",
        "inventory": "库存系统设计：采用事件溯源模式记录每次库存变更。预占-确认-释放两阶段提交防止超卖。库存预警阈值可配置，低库存自动触发补货单。",
        "库存": "库存系统设计：采用事件溯源模式记录每次库存变更。预占-确认-释放两阶段提交防止超卖。库存预警阈值可配置，低库存自动触发补货单。",
        "ui": "UI/UX设计：已输出Figma设计稿，包含移动端和桌面端双布局。组件库基于Design System v3.0，无障碍(WCAG AA)合规检查已通过。用户旅程关键节点已标注埋点。",
        "integration": "集成测试方案：上下游接口契约测试覆盖全部12个服务边界。端到端测试场景包括正常流程、异常回滚、并发冲突和超时重试。测试数据工厂已就绪。",
        "集成": "集成测试方案：上下游接口契约测试覆盖全部12个服务边界。端到端测试场景包括正常流程、异常回滚、并发冲突和超时重试。测试数据工厂已就绪。",
        "documentation": "文档已完成：包含系统架构图、API参考、部署手册和运维Runbook。所有文档采用Markdown格式存储于项目仓库docs/目录，CI自动发布到内部文档站点。",
        "文档": "文档已完成：包含系统架构图、API参考、部署手册和运维Runbook。所有文档采用Markdown格式存储于项目仓库docs/目录，CI自动发布到内部文档站点。",
        "coordinator": "协调报告：各团队交付物已收齐。依赖关系已全部解除，无阻塞项。质量门禁全部通过，具备发布条件。建议进入发布评审阶段。",
        "协调": "协调报告：各团队交付物已收齐。依赖关系已全部解除，无阻塞项。质量门禁全部通过，具备发布条件。建议进入发布评审阶段。",
        "incident_commander": "事件指挥：已确认故障等级P1，影响范围：支付服务30%请求超时。已拉起war room，on-call工程师已就位。当前排查方向：数据库连接池耗尽。预计恢复时间：30分钟。",
        "incident": "事件分析：根因定位为Redis Cluster主节点故障转移期间Session数据丢失，导致认证服务返回403。已执行手动Failover，连接池已恢复。时间线已记录。",
        "incident_responder": "响应执行：已重启受影响服务实例，健康检查通过。已向status page更新事件状态。监控告警已确认恢复。事后复盘文档已创建。",
    }

    @classmethod
    def _mock_response(cls, system_prompt: str, user_prompt: str) -> str:
        lower_sp = system_prompt.lower()
        lower_up = user_prompt.lower()
        combined = lower_sp + " " + lower_up

        # Walk the keyword table; order matters for overlapping keys.
        for keyword, response in cls._MOCK_TABLE.items():
            if keyword in combined:
                return response

        return "任务已完成。输出已结构化，可供下游Agent直接引用。关键结论已标注，边界条件已说明。"
