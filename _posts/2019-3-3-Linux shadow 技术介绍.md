---
layout:     post
title:      Linux 账户密码存储技术介绍
#subtitle:  
date:       2019-3-3
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - Linux
    - shadow
---

> [我的博客](http://feizaipp.github.io)

<p style="text-indent:2em">
在早期的 Linux 版本中，账户名和密码的密文被保存在同一个文件里，即 /etc/passwd 文件中。因为 /etc/passwd 文见对于任何用户都是可见的，所以这样做存在安全问题。黑客通过暴力破解或者字典攻击等技术手段能够破解管理员账户的密码。一旦管理员的账户密码被破解，后果可想而知。
</p>
<p style="text-indent:2em">
后来有了 shadow 技术，账户信息保存到 /etc/passwd 文件，而账户密码被加密后存储到 /etc/shadow 文件中。并且 /etc/shadow 文件被强制访问控制保护。目前几乎所有的 Linux 发行版本的账户密码存储技术都使用 shadow 技术。 Linux系统还保留了两个系统工具用来打开和关闭 shadow ，即 pwcon 和 pwuncon 。
</p>

# 1. 账户密码文件访问
<p style="text-indent:2em">
我们在什么情况下需要访问账户密码文件呢？首先，在应用程序中通过 Glibc 接口获得 passwd 或 spwd 结构体，用来获取账户信息；其次，我们登陆系统时，要验证账户的合法性以及密码的正确性，其实这里最终也是通过 Glibc 接口访问账户密码文件；最后是一些系统工具，直接通过 io 操作账户密码文件。
</p>
综上所述，账户密码文件的访问框架如下图所示：  

![账户密码文件的访问框架图](/img/pw-shadow-io1.png)

# 2.密码加密实现
<p style="text-indent:2em">
上一章介绍到，密码是被加密后存储到 /etc/shadow 文件中的，那么密码是何时被加密的，又是用什么加密算法加密的？首先，密码肯定是在写入 /etc/shadow 文件之前被加密，在设置或者修改账户密码时会将密码加密并写入 /etc/shadow 文件，所以密码是在被设置或修改的时候加密的。那么我们要了解密码加密的实现就可以顺着系统工具 passwd 程序来分析密码加密流程；其次，如果你了解 PAM 的话，应该看到过下面的 PAM 配置语句，这条配置语句中的 sha512 意味着加密算法使用的是 sha512 摘要算法。
</p>
> password sufficient pam_unix.so sha512 shadow nullok try_first_pass use_authtok  
<p style="text-indent:2em">
passwd 应用程序修改密码时，首先通过 PAM 模块对密码的合法性进行检查，检查确定密码合法后， PAM 模块调用 Glibc 的 crypt 的接口，请求对密码加密， crypt 接口中通过使用 USE_NSS 宏来控制是调用 Glibc 本地的加密模块还是调用 NSS 模块的加密模块，默认情况下宏 USE_NSS 是打开的，所以  crypt 接口调用 NSS 模块中的加密算法。 crypt 接口中传入要使用的加密算法的 id ，如下表所示。加密完成后将米文返回给 PAM 模块，并写入 /etc/shadow 文件中。
</p>
系统支持的加密算法表：  

算法 id |算法名称  
|--|--|
$1$|MD5
$2a$|BLOWFISH
$5$|SHA256
$6$|SHA512

密码加密流程如下图所示：

![密码加密流程图](/img/pw-shadow-io3.png)
