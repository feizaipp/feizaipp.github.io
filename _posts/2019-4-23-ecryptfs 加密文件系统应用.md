---
layout:     post
title:      eCryptfs 加密文件系统应用
#subtitle:  
date:       2019-4-23
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - Linux
    - eCryptfs
    - Dbus
---

> [我的博客](http://feizaipp.github.io)

# 0. 声明
&#160; &#160; &#160; &#160;本文引用了 [IBM Developer](https://www.ibm.com/developerworks/cn/linux/l-cn-ecryptfs) 文章里的部分内容。如有侵权，请通过邮箱联系我 (zpehome@yeah.net) 。

# 1. 概述
<p style="text-indent:2em">
在这个信息爆炸式增长的社会，数据的安全性越来越来受到人们的重视。常用的保护用户数据的方法是对数据加密，使用的时候对数据解密。在具体应用中又分为以下两种，一种是文件加密后，只有信任的应用程序能够访问，代表方案就是亿赛通；另一种是系统对整个磁盘进行加密，开机时通过密码或硬件设备(TPM芯片)获取秘钥，并对磁盘解密，代表方案是全盘加密。</p>
<p style="text-indent:2em">
以上两种方案存在如下问题，方案一只对指定文件格式的内容加密，如果亿赛通软件没有设置对 .mak 文件进行加密的话，那么把源码内容存储为 .mak 文件，亿赛通软件不会加密，这就造成了源码的泄露。方案二采用透明加解密技术，一旦系统获取了秘钥，任何对磁盘文件的访问都是明文的，也就是说磁盘文件可以随时以明文状态拷贝出来，造成数据泄露。</p>

<p style="text-indent:2em">
针对以上问题，设计一种以目录为单位的加解密方案，用户将私密数据放到目录中，并对数据加密。该方案利用 eCryptfs 文件系统技术，当目录以 eCryptfs 文件系统挂载时，对目录里的数据解密，当卸载掉该目录的 eCryptfs 文件系统时，对目录里的数据加密。我们可以把挂载和卸载 eCryptfs 文件系统的操作形象的比喻成打开和关闭保险箱，所以本方案称为安全保密箱。
</p>

# 2. eCryptfs 介绍
## 2.1. 加密文件系统
<p style="text-indent:2em">
加密文件系统通过将加密服务集成到文件系统这一层面来解决数据泄露问题。文件内容经过加密算法加密后以密文的形式存放在物理介质上，即使文件丢失或被窃取，在加密密钥未泄漏的情况下，文件内容也不会被泄露，从而保证了数据的安全性。与此同时，数据所有者对加密文件的访问则非常方便。用户通过身份认证后，对加密文件的访问和普通文件没有什么区别，就好像该文件并没有被加密过，这是因为加密文件系统在内核层做了相关的加密和解密的工作。由于加密文件系统工作在内核态，安全级别更高，更难以破解。内核还支持基于块设备的加密方案，就是上文提到的全盘加密，与它相比，加密文件系统具有更多的优势，例如：
</p>

* 支持文件粒度的加密
* 无需预先保留足够的空间，用户可以随时加密或恢复文件
* 对单个加密文件更改密钥和加密算法比较容易
* 不同的文件可以使用不同的加密算法和密钥，增大了破解的难度
* 只有加密文件才需要特殊的加密/解密处理，普通文件的存取没有额外开销
* 加密文件转移到别的物理介质上时，没有额外的加密/解密开销

## 2.2.	eCryptfs 简介
<p style="text-indent:2em">
eCryptfs - Enterprise Cryptographic Filesystem ，可翻译为企业级文件加密系统。eCryptfs 是在 Linux 内核 2.6.19 版本中引入的一个功能强大的企业级加密文件系统，堆叠在其它文件系统之上（如 Ext2, Ext3, ReiserFS, JFS 等），为应用程序提供透明、动态、高效和安全的加密功能。
</p>
<p style="text-indent:2em">
本质上，eCryptfs 插在 VFS （虚拟文件系统层）和下层物理文件系统之间，充当一个过滤器的角色。用户应用程序对加密文件的写请求，经系统调用层到达 VFS 层，VFS 转给 eCryptfs 文件系统组件处理，处理完毕后，再转给下层物理文件系统；读请求（包括打开文件）流程则相反。
</p>
<p style="text-indent:2em">
eCryptfs 分两步来加密单个文件， eCryptfs 先使用一种对称密钥加密算法来加密文件的内容，推荐使用 AES-128 算法，密钥 FEK（File Encryption Key）随机产生。有些加密文件系统为多个加密文件或整个系统使用同一个 FEK（甚至不是随机产生的），这会损害系统安全性，因为： 1) 如果 FEK 泄漏，多个或所有的加密文件将被轻松解密；2) 如果部分明文泄漏，攻击者可能推测出其它加密文件的内容； 3) 攻击者可能从丰富的密文中推测 FEK。
</p>
<p style="text-indent:2em">
显然 FEK 不能以明文的形式存放，因此 eCryptfs 使用用户提供的口令（Passphrase）、公开密钥算法（如 RSA 算法）或 TPM（Trusted Platform Module）的公钥来加密保护上文提及的 FEK。如果使用用户口令，则口令先被散列函数处理，然后再使用一种对称密钥算法加密 FEK。口令/公钥称为 FEFEK（File Encryption Key Encryption Key），加密后的 FEK 则称为 EFEK（Encrypted File Encryption Key）。由于允许多个授权用户访问同一个加密文件，因此 EFEK 可能有多份。
</p>
<p style="text-indent:2em">
这种综合的方式既保证了加密解密文件数据的速度，又极大地提高了安全性。虽然文件名没有数据那么重要，但是入侵者可以通过文件名获得有用的信息或者确定攻击目标，因此，最新版的 eCryptfs 支持文件名的加密。
</p>

## 2.4.	eCryptfs 的架构
<p style="text-indent:2em">
eCryptfs 加密文件系统的架构如下图所示：
</p>

![eCryptfs 加密文件系统架构](/img/ecryptfs1.png)

<p style="text-indent:2em">
eCryptfs Layer 是一个内核文件系统模块，但是没有实现在物理介质上存取数据的功能。在 eCryptfs Layer 自己的数据结构中，加入了指向下层文件系统数据结构的指针，通过这些指针，eCryptfs 就可以存取加密文件。
</p>
<p style="text-indent:2em">
Keystore 和用户态的 eCryptfs Daemon 进程一起负责密钥管理的工作。eCryptfs Layer 首次打开一个文件时，通过下层文件系统读取该文件的头部元数据，交与 Keystore 模块进行 EFEK（加密后的 FEK）的解密。前面已经提及，因为允许多人共享加密文件，头部元数据中可以有一串 EFEK。EFEK 和相应的公钥算法/口令的描述构成一个鉴别标识符，由 eCryptfs_auth_tok 结构表示。Keystore 依次解析加密文件的每一个 eCryptfs_auth_tok 结构：首先在所有进程的密钥链（key ring）中查看是否有相对应的私钥/口令，如果没有找到，Keystore 则发一个消息给 eCryptfs Daemon，由它提示用户输入口令或导入私钥。第一个被解析成功的 eCryptfs_auth_tok 结构用于解密 FEK。如果 EFEK 是用公钥加密算法加密的，因为目前 Kernel Crypto API 并不支持公钥加密算法，Keystore 必须把 eCryptfs_auth_tok 结构发给 eCryptfs Daemon，由它调用 Key Module API 来使用 TPM 或 OpenSSL库解密 FEK。解密后的 FEK 以及加密文件内容所用的对称密钥算法的描述信息存放在 eCryptfs_inode_info 结构的 crypt_stat 成员中。eCryptfs Layer 创建一个新文件时，Keystore 利用内核提供的随机函数创建一个 FEK；新文件关闭时，Keystore 和 eCryptfs Daemon 合作为每个授权用户创建相应 EFEK ，并存放在加密文件的头部元数据中。
</p>
<p style="text-indent:2em">
eCryptfs 采用 OpenPGP 的文件格式存放加密文件，详情参阅 RFC 2440 规范。我们知道，对称密钥加密算法以块为单位进行加密/解密，例如 AES 算法中的块大小为 128 位。因此 eCryptfs 将加密文件分成多个逻辑块，称为 extent。当读入一个 extent 中的任何部分的密文时，整个 extent 被读入 Page Cache，通过 Kernel Crypto API 被解密；当 extent 中的任何部分的明文数据被写回磁盘时，需要加密并写回整个 extent 。 extent 的大小是可调的，但是不会大于物理页的尺寸。当前的版本中的 extent 默认值等于物理页大小，因此在 IA32 体系结构下就是 4096 字节。加密文件的头部存放元数据，包括元数据长度、标志位以及 EFEK 链，目前元数据的最小长度为 8192 字节。
eCryptfs 加密/解密操作流程图如下：
</p>

![eCryptfs 加密/解密操作流程图](/img/ecryptfs2.png)

## 2.5.	eCryptfs 的缺陷
<p style="text-indent:2em">
eCryptfs 第一个缺陷在于写操作性能比较差。用 iozone (主要用来测试操作系统文件系统性能的测试工具)测试了 eCryptfs 的性能，发现读操作的开销不算太大，有些小文件测试项目反而性能更好；对于写操作，所有测试项目的结果都很差。这是因为 Page Cache 里面只存放明文，因此首次数据的读取需要解密操作，后续的读操作没有开销；而每一次写 x 字节的数据，就会涉及 ((x – 1) / extent_size + 1) * extent_size 字节的加密操作，因此开销比较大。
</p>
<p style="text-indent:2em">
另外，有两种情况可能造成信息泄漏。首先，当系统内存不足时， Page Cache 中的加密文件的明文页可能会被交换到 swap 区，目前的解决方法是用 dm-crypt 加密 swap 区。其次， eCryptfs 实现的安全性完全依赖于操作系统自身的安全。如果 Linux Kernel 被攻陷，那么黑客可以轻而易举地获得文件的明文，FEK 等重要信息。
</p>

# 3. CentOS 系统集成 eCryptfs
## 3.1.	适配内核
<p style="text-indent:2em">
CentOS 7.4 系统，内核版本 kernel-3.10 ，默认情况下 CentOS 7.4 系统的内核不支持 eCryptfs 文件系统，需要开启内核选项。
</p>
&#160; &#160; &#160; &#160;CentOS 7.4 系统的内核的 eCryptfs 模块有 bug ，我从 CentOS Bug Tracker 论坛上找到解决方案，经测试可以解决此 Bug 。这里是相关 [Bug](https://bugs.centos.org/view.php?id=15353) 的链接。我把相关的 patch 放到我的 [github](https://github.com/feizaipp/ecryptfs-app/tree/master/patch) 中去了，也可以去那里下载。

&#160; &#160; &#160; &#160;ecryptfs-utils 提供了 eCryptfs 文件系统应用层的开发库和帮助工具。可以从 github 上下载最新版的 [eCryptfs-uitls](https://github.com/dustinkirkland/ecryptfs-utils) 。

# 4. eCryptfs 应用
<p style="text-indent:2em">
eCryptfs 文件系统的应用非常简单，将想要存放加密数据的目录挂载成 eCryptfs 文件系统类型，例如执行 sudo mount -t eCryptfs real_path eCryptfs_mounted_path 命令。推荐 eCryptfs_mounted_path 和 真实目录 real_path 一致，这样非授权用户不能通过原路径访问加密文件。使用完成后，执行 umount eCryptfs_mounted_path 命令，此时该目录的数据都被加密了。基于 eCryptfs 文件系统，我们可以提供给用户管理自己隐私数据的系统，这里就暂且称它为数据保险箱，提供创建、打开、关闭数据保险箱的功能。创建保险箱时，提供用户输入该保险箱的密码；打开保险箱时用户需要输入正确的密码，打开成功后，用户可以读写数据保险箱里的数据，此时的数据对于用户来说是透明的，已经经过 eCryptfs 文件系统解密；关闭保险箱后，用户再次访问数据就是密文数据了。
</p>

<p style="text-indent:2em">
由于 mount 命令只有 root 账户才能执行，所以要想所有用户使用数据保险箱的话，在数据保密箱的软件架构上应该采用 C/S 架构，即实现后台服务进程和客户端进程。其中，后台服务进程以 root 权限执行；客户端进程通过 DBus 总线与后台服务进程通信，并提供给所有用户使用。
</p>

## 4.1. 创建数据保险箱

![创建数据保险箱](/img/ecryptfs3.png)
## 4.2. 打开数据保险箱

![打开数据保险箱](/img/ecryptfs4.png)
## 4.3. 关闭数据保险箱

![关闭数据保险箱](/img/ecryptfs5.png)
