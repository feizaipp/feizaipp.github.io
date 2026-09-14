const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../js/journal.js'), 'utf8');

// Exercise the loader against browser events without contacting or posting to GitHub.
function setup(search = "") {
    class Element {
        constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.listeners = {}; this.hidden = false; }
        set textContent(value) { this.text = value; this.children = []; }
        get textContent() { return this.text || ''; }
        appendChild(child) { this.children.push(child); return child; }
        setAttribute(name, value) { this.attrs[name] = value; }
        addEventListener(name, cb) { this.listeners[name] = cb; }
        querySelector(tag) { return this.children.find(child => child.tag === tag) || null; }
    }
    const mount = new Element('div'), status = new Element('p'), retry = new Element('button');
    const listeners = new Set(), timers = new Map(), observers = [];
    let id = 0;
    const comments = {
        dataset: {commentsRepo: 'owner/blog', commentsTerm: '/2021/article/'},
        querySelector(selector) { return {'.comments-mount':mount, '.comments-status':status, '.comments-load':retry}[selector]; }
    };
    const window = {
        location: {search},
        addEventListener(type, cb) { if (type === 'message') listeners.add(cb); },
        removeEventListener(type, cb) { listeners.delete(cb); }
    };
    vm.runInNewContext(source, {
        window, URLSearchParams,
        document: {querySelectorAll:()=>[], querySelector:()=>comments, createElement:tag=>new Element(tag)},
        MutationObserver: class {constructor(cb) {this.cb=cb; observers.push(this);} observe() {} disconnect() {}},
        setTimeout(cb) {timers.set(++id,cb); return id;},
        clearTimeout(id) {timers.delete(id);}
    });
    function start() { retry.listeners.click(); return mount.children[0]; }
    function frame(host) {
        const iframe = new Element('iframe'); iframe.contentWindow = {};
        host.appendChild(iframe); observers.at(-1).cb(); return iframe;
    }
    function message(iframe, overrides={}) {
        const event = {origin:'https://utteranc.es',source:iframe.contentWindow,data:{type:'resize',height:320},...overrides};
        [...listeners].forEach(cb=>cb(event));
    }
    return {mount,status,retry,start,frame,message,listeners,timers,expire:()=>[...timers.values()].forEach(cb=>cb())};
}

test('passes stable article mapping and waits for widget readiness', () => {
    const h=setup(), host=h.start(), script=host.children[0], frame=h.frame(host);
    assert.equal(script.attrs.repo,'owner/blog');
    assert.equal(script.attrs['issue-term'],'/2021/article/');
    assert.equal(frame.loading,'eager');
    assert.equal(h.mount.attrs['aria-busy'],'true');
    assert.equal(h.status.hidden,false); // Creating/loading an iframe is not readiness.
    h.message(frame);
    assert.equal(h.status.hidden,true);
    assert.equal(h.mount.attrs['aria-busy'],'false');
    assert.equal(h.listeners.size,0);
    assert.equal(h.timers.size,0);
});

test('rejects unrelated origins, frames and invalid resize events', () => {
    const h=setup(), f=h.frame(h.start());
    for (const override of [{origin:'https://example.com'},{source:{}},{data:{type:'resize',height:0}},{data:{type:'resize',height:Infinity}},{data:{type:'resize',height:'320'}},{data:{type:'other',height:320}}]) {
        h.message(f,override); assert.equal(h.status.hidden,false);
    }
    h.message(f); assert.equal(h.status.hidden,true);
});

test('slow response recovers after timeout without another click', () => {
    const h=setup(), f=h.frame(h.start()); h.expire();
    assert.equal(h.retry.hidden,false);
    assert.match(h.status.textContent,/暂时无法加载/);
    h.message(f); assert.equal(h.status.hidden,true); assert.equal(h.retry.hidden,true);
});

test('retry isolates stale scripts and messages from the previous attempt', () => {
    const h=setup(), oldHost=h.start(), oldFrame=h.frame(oldHost);
    const oldScript=oldHost.children[0]; oldScript.onerror();
    const current=h.start(), currentFrame=h.frame(current);
    assert.notEqual(current,oldHost);
    assert.equal(h.mount.children.length,1);
    oldScript.onerror(); h.message(oldFrame);
    assert.equal(h.mount.attrs['aria-busy'],'true');
    assert.equal(h.status.hidden,false);
    h.message(currentFrame); assert.equal(h.status.hidden,true);
});

test('ignores duplicate clicks while loading and after readiness', () => {
    const h=setup(), first=h.start(); h.start();
    assert.equal(h.mount.children[0],first);
    h.message(h.frame(first)); h.start(); assert.equal(h.mount.children[0],first);
});

test('sign-in callback starts loading without waiting for scrolling', () => {
    const h=setup('?utterances=test-only-placeholder');
    assert.equal(h.mount.children.length,1);
    assert.equal(h.mount.attrs['aria-busy'],'true');
    assert.equal(h.mount.children[0].children[0].attrs['issue-term'],'/2021/article/');
});
