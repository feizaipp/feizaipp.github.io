## 本地预览

安装 Jekyll 和 `jekyll-paginate` 后，在项目目录运行：

```sh
LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 jekyll serve
```

打开 http://localhost:4000。UTF-8 环境用于正确处理中文文件名。

## 外观维护

- `css/journal.css`：配色、排版，以及桌面和手机样式；无需重新编译 LESS。
- `index.html`：首页介绍、文章列表和主题导航；文章数与主题数自动生成。
- `_includes/post-excerpt.html`：优先使用文章的 `description`，否则提取正文段落作为摘要。
- `_includes/nav.html`：导航和手机菜单。

现有文章仍放在 `_posts`，保留 Jekyll 分页与文章链接规则。

## 代码复制与评论

文章中的代码块自动提供“复制”按钮，复制原文并保留缩进和换行。优先使用浏览器 Clipboard API，权限受限或 HTTP 页面会尝试兼容方式，并显示成功或失败状态。

评论使用 [Utterances](https://utteranc.es/)，评论内容保存在 GitHub Issues 中。本仓库已于 2026-09-15 完成 Utterances 安装，仅授权 `feizaipp/feizaipp.github.io` 的元数据读取及 Issues 读写。迁移到其他仓库时：

1. 确保 `feizaipp/feizaipp.github.io` 是公开仓库并启用了 Issues。
2. 在 [Utterances GitHub App](https://github.com/apps/utterances) 安装页选择该仓库。
3. 发布博客后，打开任意文章底部，使用 GitHub 登录即可留言。首条评论会自动创建对应 Issue。

本地预览请使用 `jekyll serve`；若使用静态预览服务器，应在仅用于本地的 Jekyll 配置中将 `url` 设为预览地址，使 GitHub 登录后返回本地页面。正式站点的 `url` 保持 HTTPS 线上地址。

运行 `npm test` 可验证评论就绪识别、错误消息过滤、超时恢复及重试隔离。

`_config.yml` 的 `comments.repo` 设置评论仓库；`comments.enabled: false` 可关闭全站评论。文章 front matter 中设置 `comments: false` 可单独关闭。评论按文章路径绑定，改标题不会创建新讨论；更改文章路径会影响绑定。

评论接近可视区域时自动加载；收到评论组件的就绪消息后结束加载提示。30 秒未就绪时提供重试和 GitHub 链接，迟到的响应也可自动恢复。不开启 JavaScript 也能通过 GitHub 链接访问讨论。

旧 Gitalk/Disqus 嵌入已移除。旧评论仍在原服务中，不会自动迁移到 Utterances；原仓库 Issues 可继续查阅。旧 Gitalk OAuth 密钥曾存在于公开配置和页面，请在 GitHub OAuth App 设置中撤销或重新生成；仅删除当前文件不会清除历史记录。

## 致谢

1. 这个模板是从这里 [BY](https://github.com/qiubaiying/qiubaiying.github.io) fork 的, 感谢这个作者。 
2. 感谢 Jekyll、Github Pages 和 Bootstrap!

## License

遵循 MIT 许可证。有关详细,请参阅 [LICENSE](https://github.com/feizaipp/feizaipp.github.io/blob/master/LICENSE)。

