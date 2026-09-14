/* Reading tools: progressive enhancement, no framework dependency. */
(function () {
    'use strict';

    function legacyCopy(text) {
        var active = document.activeElement;
        var selection = window.getSelection();
        var ranges = [];
        for (var i = 0; selection && i < selection.rangeCount; i++) {
            ranges.push(selection.getRangeAt(i).cloneRange());
        }
        var input = document.createElement('textarea');
        input.value = text;
        input.setAttribute('readonly', '');
        input.className = 'copy-buffer';
        document.body.appendChild(input);
        input.select();
        var copied = false;
        try { copied = document.execCommand('copy'); } catch (error) { copied = false; }
        input.remove();
        if (active && active.focus) active.focus({ preventScroll: true });
        if (selection) {
            selection.removeAllRanges();
            ranges.forEach(function (range) { selection.addRange(range); });
        }
        return copied;
    }

    document.querySelectorAll('.post-container pre').forEach(function (pre, index) {
        var code = pre.querySelector('code') || pre;
        var wrapper = document.createElement('div');
        wrapper.className = 'code-block';
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);

        var toolbar = document.createElement('div');
        toolbar.className = 'code-toolbar';
        var language = document.createElement('span');
        var languageNode = code.closest('[class*="language-"]') || pre.closest('[class*="language-"]');
        var match = languageNode && languageNode.className.match(/(?:^|\s)language-([\w+-]+)/);
        language.textContent = match ? match[1] : 'CODE';
        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'copy-code';
        button.textContent = '复制';
        button.setAttribute('aria-label', '复制第 ' + (index + 1) + ' 段代码');
        var status = document.createElement('span');
        status.className = 'copy-status';
        status.setAttribute('role', 'status');
        status.setAttribute('aria-live', 'polite');
        toolbar.appendChild(language);
        toolbar.appendChild(status);
        toolbar.appendChild(button);
        wrapper.insertBefore(toolbar, pre);
        var timer;
        button.addEventListener('click', async function () {
            clearTimeout(timer);
            button.disabled = true;
            var text = code.textContent;
            var copied = false;
            try {
                if (navigator.clipboard && window.isSecureContext) {
                    await navigator.clipboard.writeText(text);
                    copied = true;
                }
            } catch (error) { /* Fall back when clipboard permission is unavailable. */ }
            if (!copied) copied = legacyCopy(text);
            button.disabled = false;
            button.textContent = copied ? '已复制 ✓' : '重试';
            status.textContent = copied ? '代码已复制' : '复制失败，请选中代码手动复制';
            button.classList.toggle('is-copied', copied);
            timer = setTimeout(function () {
                button.textContent = '复制';
                button.classList.remove('is-copied');
                status.textContent = '';
            }, 3000);
        });
    });

    var comments = document.querySelector('[data-comments-repo]');
    if (!comments) return;
    var mount = comments.querySelector('.comments-mount');
    var status = comments.querySelector('.comments-status');
    var retry = comments.querySelector('.comments-load');
    var state = 'idle';
    var attempt = 0;
    var cancelAttempt = function () {};

    function loadComments() {
        if (state === 'loading' || state === 'loaded') return;
        cancelAttempt();
        state = 'loading';
        var currentAttempt = ++attempt;
        mount.textContent = '';
        // Isolate each attempt so a late script cannot replace the current thread.
        var host = document.createElement('div');
        mount.appendChild(host);
        retry.hidden = true;
        status.hidden = false;
        status.textContent = '正在加载评论…';
        mount.setAttribute('aria-busy', 'true');
        var timeout;
        var observer;

        function cleanup() {
            clearTimeout(timeout);
            observer.disconnect();
            window.removeEventListener('message', onMessage);
        }
        cancelAttempt = cleanup;
        function fail() {
            if (currentAttempt !== attempt || state !== 'loading') return;
            state = 'error';
            clearTimeout(timeout);
            // Keep listening: a slow connection may still finish without a retry.
            status.textContent = '评论暂时无法加载。可以重试，或前往 GitHub 参与讨论。';
            retry.textContent = '重新加载';
            retry.hidden = false;
            mount.setAttribute('aria-busy', 'false');
        }
        function onMessage(event) {
            var frame = host.querySelector('iframe');
            var data = event.data;
            // Utterances reports its rendered height once the comment UI is ready.
            // An iframe load event alone can also mean a blank or failed document.
            if (currentAttempt !== attempt || event.origin !== 'https://utteranc.es' ||
                !frame || event.source !== frame.contentWindow || !data ||
                data.type !== 'resize' || typeof data.height !== 'number' ||
                !Number.isFinite(data.height) || data.height <= 0) return;
            state = 'loaded';
            cleanup();
            status.hidden = true;
            retry.hidden = true;
            mount.setAttribute('aria-busy', 'false');
        }
        observer = new MutationObserver(function () {
            var frame = host.querySelector('iframe');
            if (!frame) return;
            observer.disconnect();
            frame.title = '文章评论（GitHub 登录）';
            frame.loading = 'eager';
        });
        observer.observe(host, { childList: true, subtree: true });
        window.addEventListener('message', onMessage);
        var script = document.createElement('script');
        script.src = 'https://utteranc.es/client.js';
        script.setAttribute('repo', comments.dataset.commentsRepo);
        // Use the canonical article path, not a local preview query or hash.
        script.setAttribute('issue-term', comments.dataset.commentsTerm);
        script.setAttribute('theme', 'github-light');
        script.setAttribute('crossorigin', 'anonymous');
        script.async = true;
        script.onerror = fail;
        timeout = setTimeout(fail, 30000);
        host.appendChild(script);
    }

    retry.addEventListener('click', loadComments);
    if (new URLSearchParams(window.location.search).has('utterances')) {
        // Consume the sign-in callback immediately, even before the reader scrolls.
        loadComments();
    } else if ('IntersectionObserver' in window) {
        var visibility = new IntersectionObserver(function (entries) {
            if (entries.some(function (entry) { return entry.isIntersecting; })) {
                visibility.disconnect();
                loadComments();
            }
        }, { rootMargin: '250px' });
        visibility.observe(comments);
    }
})();
