---
layout:     post
title:      SEAndroid 介绍
#subtitle:  
date:       2019-3-28
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - SELinux
    - SEAndroid
---

> [我的博客](http://feizaipp.github.io)

# 1. 介绍
<p style="text-indent:2em">
SEAndroid 是 Google 在 Android 4.4 上正式推出的一套以 SELinux 为基础核心的系统安全机制。 SELinux 是针对 Linux 设计的安全加强系统。
</p>

# 2. SELinux 背景知识
<p style="text-indent:2em">
Google 针对 SELinux 进行了一定的修改，从而得到了 SEAndroid ，所以我们先了解下 SELInux 相关知识。
</p>

## 2.1.	DAL 和 MAC
<p style="text-indent:2em">
SELinux 出现之前， Linux 上的安全模型叫 DAC ，全称是 Discretionary Access Control ，翻译为自主访问控制。 DAC 的核心思想很简单，就是：进程理论上所拥有的权限与执行它的用户的权限相同。比如，以 root 用户启动 Browser ，那么 Browser 就有 root 用户的权限，在 Linux 系统上能干任何事情。
</p>
<p style="text-indent:2em">
显然， DAC 太过宽松了。 SELinux 为了解决这个问题设计了一个新的安全模型，叫 MAC(Mandatory Access Control) ，翻译为强制访问控制。 MAC 的核心思想是：任何进程想在 SELinux 系统中干任何事情，都必须先在安全策略配置文件中赋予权限，凡是没有出现在安全策略配置文件中的权限，进程都没有该权限。例如：在 external/sepolicy/netd.te 中有下面语句： allow netd proc:file write ，该语句表示允许 netd 域中的进程写类型为 proc 的文件。如果在 net.te 中没有使用以上语句，则 netd 就无法往 /proc 目录下的任何文件中写数据，即使 netd 具有 root 权限。
</p>

## 2.2.	SELinux Policy 语言简介
<p style="text-indent:2em">
通过以上的介绍，在 SELinux 中，安全策略文件是最重要的，所以了解 SELinux Policy 语言是很有必要的。
</p>
<p style="text-indent:2em">
Linux 中有两种东西，文件和进程。进程能发起动作，例如它能打开文件并操作它，而文件只能被进程操作。 SELinux 中，每种东西都会被赋予一个安全属性，官方说法叫 Security Context(SContext) ， SContext 是一个字符串。
</p>
<p style="text-indent:2em">
在 SEAndroid 中，进程的 SContext 可通过 ps –Z 命令查看，如下图所示：
</p>

![进程的安全标签](/img/process_scontext.png)

<p style="text-indent:2em">
上图中最左侧的那一列是进程 SContext 。以第一个进程 /init 为例，其值为 u:r:init:s0 ，其中：
</p>

* u 为 user 的意思。 SEAndroid 中定义了一个 SELinux 用户，值为 u ；
* r 为 role 的意思。 role 是角色的意思，它是 SELinux 中一种比较高层的东西，更方便的权限管理思路，即 Role Based Access Control ，翻译为基于角色的访问控制，简称 RBAC 。一个 u 可以属于多个 role ，不同的 role 具有不同的访问权限
* init 代表该进程所属的 Domain 为 init 。 MAC 的基础管理思路是针对 Type Enforcement Access Control ，简称 TEAC ，一般用 TE 表示。对进程来说 Type 就是 Domain 。比如 init 这个 Domain 有什么权限，都需要通过 allow 语句说明， allow 语句写在安全策略配置文件里。
* s0 与 Multi-Level Security(MLS)机制有关。 MLS 将系统的进程和文件进行了分级，不同级别的资源需要对应级别的进程才能访问。

<p style="text-indent:2em">
在 SEAndroid 中，文件的的 SContext 可以通过 ls –Z 来查看，如下图所示：
</p>

![文件的安全标签](/img/file_scontext.png)

<p style="text-indent:2em">
上图中倒数第二列所示为文件和目录的 SContext 信息，以第一行 ueventd.qcom.rc 为例，其信息为 u:object_r:rootfs:s0 。
</p>

* u 表示 user，它代表创建这个文件的 SELinux 用户
* object_r 表示文件对应的 role。文件是死的东西，它没办法扮演角色，所以 SELinux 中，死的东西都用 object_r 来表示 role
* rootfs 表示文件对应的 Type，和进程的 Domain 是一个意思。它表示 ueventd.qcom.rc 文件对应的 Type 为 rootfs
* s0 表示 MLS 的级别
<p style="text-indent:2em">
根据 SELinux 规范，完整的 SContext 字符串为： user:role:type:[rang] ， rang 是可选项。
</p>

## 2.3.	TE 介绍
<p style="text-indent:2em">
前面已经介绍了 TE 的 allow 语句，再来分析一下它：
</p>

```
allow netd proc:file write
```
<p style="text-indent:2em">
这条语句的语法为：
</p>

* allow ： TE 的 allow 语句，表示授权。除了 allow 之外，还有 allowaudit 、 dontaudit 、 neveraudit 等
* netd ： source type ，也叫 subject 或 domain
* proc ： target type ，它代表其后的 file 所对应的 type
* file ：代表 Object Class ，它代表能够给 subject 操作的一类东西。例如 file 、 dir 、 socket 等。 Android 系统存在其他 Linux 系统没有的 Object Class ，即 binder 和 zygote
* write ：在该类Object Class中所定义的操作

<p style="text-indent:2em">
根据SELinux规范，完整的allow相关的语句格式为：
</p>

```
rule_name source_type target_type:object_class perm_set
```

* 多个 perm_set 用 {} 括起来，中间用空格隔开，例如， allow zygote appdomain:dir { getattr search }; 表示允许 zygote 域中的进程 search 或 getattr 类型为 appdomain 的目录
* allow unconfineddomain {fs_type dev_type file_type}:{ chr_file file }   ~{entrypoint relabelto}; source_type 为 unconfineddomain ， target_type 为一组 type ，由 {fs_type dev_type file_type} 构成， object_class 也包含两个，为 { chr_file file } ， perm_set 语法比较奇特，前面有一个 '~' 号。它表示出了 {entrypoint relabelto} 之外， { chr_file file } 这两个 object_class 所拥有的其他操作
* 特殊符号出了 '~' 之外，还有-号和 '\*' 号，其中 '-' 号表示去除某项内容， '*' 号表示所有内容
* neverallow { appdomain -unconfineddomain } self:capability2 *; 这条语句中， source_type 为 appdomain ，但不属于 unconfineddomain 的进程，而 * 号表示所有和 capability2 相关的权限， neverallow 表示决不允许
* 特别注意，权限必须显示声明，没有声明的话默认就没有权限。那 neverallow 语句就没有必要存在了，因为无权限是不需要声明的。确实如此， neverallow 语句的作用只是生成安全策略文件时进行检查，判断是否有违法 neverallow 语句的 allow 语句

### 2.3.1. rule_name
<p style="text-indent:2em">
TE 中的 rule_name 有四种：
</p>

* allow ：赋予某项权限
* allowaudit ： audit 含义是记录某项操作，默认情况下 SELinux 只记录那些权限检查失败的操作， allowaudit 则使得权限检查成功的操作也被记录
* dontaudit ：对那些权限检查失败的操作不做记录
* neverallow ：用来检查安全策略文件中是否有违反该项规则的 allow 语句

### 2.3.2. Object Class
<p style="text-indent:2em">
根据 SELinux 的规范， Object Class 类型由 class 关键字声明。 Android 平台支持的 Object Class 在 external/sepolicy/security_classes
</p>

```
# file-related classes
class filesystem
class file  #代表普通文件
class dir   #代表目录
class fd    #代表文件描述符
class lnk_file  #代表链接文件
class chr_file  #代表字符设备文件
# network-related classes
class socket   #socket
class tcp_socket
class udp_socket
class binder   #Android 平台特有的 binder
class zygote   #Android 平台特有的 zygote
# Android 平台特有的属性服务。其后的 userspace 这个词和用户空间中的 SELinux 权限检查有关
class property_service 	# userspace
```

### 2.3.3. Perm Set
<p style="text-indent:2em">
根据 SELinux 规范， Perm Set 也需要声明，在 external/sepolicy/access_vectors 文件中。
</p>

<p style="text-indent:2em">
SELinux 规范中，定义 Perm Set 有两种方式，一种是使用下面的 common 命令，其格式为： common common_name {permission_name…} 。另一种是使用 class 命令，其格式为： class class_name [ inherits common_name ] { permission_name ... } ， inherits 表示继承了某个 common 定义的权限，注意 class 命令定义的权限其实针对的就是某个 object class ，它不能被其他 class 继承，例如：
</p>

```
class dir inherits file {
    add_name remove_name reparent search rmdir open audit_access execmod
}
```

<p style="text-indent:2em">
来看 SEAndroid 中的 binder 和 property_service 这两个 Object class 定义了哪些操作权限：
</p>

```
class binder {impersonate  call set_context_mgr transfer}
class property_service { set }
```

### 2.3.4. type 和 attribute
<p style="text-indent:2em">
type 命令的完整格式为： type type_id [alias alias_id,] [attribute_id] ；其中方括号中的内容是可选的， alias 指定了 type 的别名，可以指定多个别名。例如： type shell,domain 定义了一个名为 shell 的 type ，它和一个名为 domain 的属性关联，可以关联多个属性。属性由 attribute 关键字定义， attribute 文件中定义的 SEAndroid 使用的属性有： attribute domain 、 attribute file_type 等。
</p>

<p style="text-indent:2em">
可以在定义 type 的时候直接和某个 attribute 关联，也可以单独通过 typeattribute 将某个 type 和某个或多个 attribute 关联。例如： typeattribute system mlstrustedsubject 表示将 system 类型和 mlstrustedsubject 属性关联起来。 type 和 attribute 位于同一命名空间，不能用 type 命令和 attribute 命令定义相同名字的东西。 attribute 真正的意思应该是类似 type group 这样的概念，比如， type A 和 attribute B 关联起来，就是说 type A 属于 group B 中的一员。那么，使用 attribute 有什么好处呢？一般而言，系统会定义数十个或数百个 type ，每个 type 都需要通过 allow 语句来设置权限，这样安全策略文件编起来就会非常麻烦。有了 attribute 后，我们可以将这些 type 与某个 attribute 关联起来，然后用 allow 语句，直接将 source_type 设置为这个 attribute 就可以了。
</p>

## 2.4. RBAC 和 constrain
<p style="text-indent:2em">
绝大多数情况下， SELinux 的安全策略需要我们编写各种各样的 xx.te 文件。由前文可知 .te 文件内包含了各种 allow 、 type 等语句，这些都是 TEAC ，属于 SELinux MAC 中的核心组成部分。
</p>
<p style="text-indent:2em">
在TEAC之上还有一种基于 Role 的安全策略，也就是 RBAC 。 RBAC 是如何实施相关的权限控制呢？我们先来看 SEAndroid 中 Role 和 User 的定义。 Android 中只定义了一个 role ，名字是 r ，在 external/sepolicy/roles 文件中： role r; 
</p>

<p style="text-indent:2em">
将上面定义的 r 与 attribute domain 关联起来：
</p>

```
role r types domain;
```

<p style="text-indent:2em">
再来看看 user 的定义，支持 MLS 的 user 定义格式为：
</p>

```
user seuser_id roles role_id level mls_level range mls_range
```
<p style="text-indent:2em">
不支持 MLS 的 user 定义格式为：
</p>

```
user seuser_id roles role_id;
```
<p style="text-indent:2em">
在 SEAndroid 中使用了支持 MLS 的格式。在 external/sepolicy/users 定义了 user：
</p>

```
user u roles { r } level s0 range s0 – mls_systemhigh;
```
<p style="text-indent:2em">
上面定义的 user u 将和 role r 关联，一个 user 可以和多个 role 关联。 level 后的是该 user 具有的安全级别。 s0 为最低级别，也就是默认的级别， mls_systemhigh 为 u 所能获得的最高安全级别。
</p>
<p style="text-indent:2em">
那么， role 和 user 有什么样的权限控制呢？
</p>

* 允许从一个 role 切换到另一个 role ，例如： allow from_role_id to_role_id
* 角色之间的关系。 Dominance {role super_r {role sysadm_r;role secadm_r;}} 这句话表示 super_r dominance sysadm_r 和 secadm_r 这两个角色。从 type 的角度看， super_r 将自动继承 sysadm_r 和 secadm_r 所关联的 type 或 attribute

<p style="text-indent:2em">
怎样实现基于 role 和 user 的权限控制呢？
</p>
<p style="text-indent:2em">
SELinux提供了一个关键字， constrain 。 constrain 的标准格式是： constrain object_class_set perm_set expression; 
例如： constrain file write (u1==u2 and r1==r2); 表示只有 source 和 target 的 user 相同，并且 role 也相同，才允许 write object_class 为 file 的客体。 Constrain 中最关键的是 expression ，它包含如下关键词：
</p>

* u1 r1 t1 ：代表 source 的 user role 和 type
* u2 r2 t2 ：代表 target 的user role 和 type
* == 和 != ： == 表示相等或属于， != 表示不等或不属于。对于 r，u 来说， == 和 != 表示相等或不相等。而当 t1== 或 t1!= 某个 attribute 时，表示源 type 属于或不属于这个 attribute
* dom domby incomp eq ；仅仅针对 role，表示统治，被统治，没关系和相等
<p style="text-indent:2em">
SEAndroid 中没有使用 constrain，而是使用 mlsconstrain。
</p>

## 2.5.	Labeling 介绍
<p style="text-indent:2em">
上面介绍了 SELinux 中最常见的东西，那么，这些 SContext 最开始是怎么赋值这些进程和文件的呢？ SELinux 中，设置或分配 SContext 给进程或文件的工作叫 Security Labeling 。 Android 系统启动后， init 进程会将一个编译完的安全策略文件传递给 kernel 以初始化 kernel 中的 SELinux 相关的模块(姑且用 Linux Security Module:LSM 来表示它)，然后 LSM 可根据其中的信息给相关 object 打标签。
</p>

### 2.5.1. sid 和 sid_context
<p style="text-indent:2em">
LSM 初始化时所需要的信息以及 SContext 信息保存在两个特殊文件中，以 Android 为例，它们分别是： initial_sids 和 initial_sid_context 。
</p>

<p style="text-indent:2em">
initial_sids 定义了 LSM 初始化相关的信息。SID 是 SELinux 中的一个概念，全称是 Security Identifier 。SID其实类似 SContext 的 key 值。因为在实际运行时，如果总是去比较字符串(SContext 是字符串)会严重影响效率。所以 SELinux 用 SID 来匹配某个 Scontext ； initial_sid_context 为这些 SID 设置最初的 SContext 信息。
</p>

<p style="text-indent:2em">
在 external/sepolicy/initial_sids 中有如下内容：
</p>

```
sid kernel #sid 是关键字，用于定义一个 sid
```

<p style="text-indent:2em">
在 external/sepolicy/initial_sid_context 中有如下内容：
</p>

```
sid kernel u:r:kernel:s0 #将 initial_sids 中定义的 sid 和初始化的 SContext 关联起来。
```

### 2.5.2. Domain/Type Transition
<p style="text-indent:2em">
SEAndroid 中， init 进程的 SContext 为 u:r:init:s0 ，而 init 创建的子进程显然不会也不可能拥有和 init 进程一样的 SContext 。那么这些子进程是怎么打上和其父进程不一样的 SContext 的呢？ SELinux 中，上述问题被称为 Domain Transition ，即某个进程的 Domain 切换到一个更合适的 Domain 中去。 Domain Transition 也是需要我们在安全策略文件中来配置的，使用 type_transition 关键字， type_transition 的完整格式为：
</p>

```
type_transition source_type target_type:class default_type
```
<p style="text-indent:2em">
例如：
</p>

```
type_transition init_t apache_exec_t:process apache_t
```
<p style="text-indent:2em">
上面的例子解释如下：当 init_t Domain 中的进程执行 type 为 apache_exec_t 类型的可执行文件(fork 并 execv)时，其 class(此处是 process)所属的 Domain 需要切换到 apache_t 域。因为 kernel 中，从 fork 到 execv 一共设置了三处 Security 检查点，所以执行上面语句，还需要至少三个 allow 语句配合：
</p>

* 首先，得让 init_t 域中的进程能够执行 type 为 apache_exec_t 的文件 : allow init_t apache_exec_t:file execute;
* 然后，告诉 SELinux ，允许 init_t 做 Domain Transition 切换进入 apache_t 域 : allow init_t apache_t:process transition
* 最后，告诉 SELinux ，切换入口为执行 apache_exec_t 类型的文件 allow apache_t apache_exec_t:file entrypoint

### 2.5.3. 宏
<p style="text-indent:2em">
按照以上规范编写 TE 文件会比较麻烦，还好 SELinux 支持宏，这样我们可以定义一个宏语句把上述 4 个步骤全部包含进来。在SEAndroid 中，系统定义的宏全在 te_macros 文件中，其中和 Domain Transition 相关的宏定义如下：
</p>

```
# external/sepolicy/te_macros 文件中
define(‘domain_trans’, ’
allow $1 $2:file {getattr open read execute};
allow $1 $3:process transition;
allow $3 $2:file {entrypoint read execute};
allow $3 $1:process sigchld;
dontaudit $1 $3:process noatsecure;
allow $1 $3:process {signh rlimitinh};
’)
```
<p style="text-indent:2em">
上面定义了 domain_trans 宏。$1,$2…等代表宏的第一个、第二个…参数。
</p>

```
define(‘domain_auto_trans’, ‘
domain_trans($1,$2,$3)
type_transition $1 $2:process $3;
’)
```
<p style="text-indent:2em">
上面定义 domain_auto_trans 宏，这个宏才是我们在 .te 文件中直接使用的，上面的例子可以使用下面的宏： 
</p>

```
domain_auto_trans(init_t,apache_exec_t,apache_t)
```

### 2.5.4. File/File System 打 label
#### 2.5.4.1 file_contexts 文件
<p style="text-indent:2em">
在 SEAndroid 中，关于 file 的 SContext 在 external/sepolicy/file_contexts 文件中。在该文件中注意 * 号， ? 号，代表通配符， * 号代表 0 个或多个字符， ? 号代表一个字符。注意 -- 号， SELinux 中类似的符号还有：
</p>

* ‘-b’ - Block Device 
* ‘-c’ - Character Device
* ‘-d’ - Directory 
* ‘-p’ - Named Pipe
* ‘-l’ - Symbolic Link 
* ‘-s’ - Socket
* ‘--’ - Ordinary file
#### 2.5.4.2 fs_use 文件
<p style="text-indent:2em">
external/sepolicy/fs_use 文件，该文件描述了 SELinux 的 labeling 信息在不同的文件系统时的处理方式。对于常规的文件系统， SContext 信息存储在文件节点的属性中，系统可以通过 getattr 函数读取 inode 中的 SContext 信息。对于这种 labeling 方式， SELinux 定义了 fs_use_xattr 关键字。这种 SContext 是永久性的保存在文件系统中的。例如：
</p>

```
fs_use_xattr yaffs2 u:object_r:labeledfs:s0;
fs_use_xattr jffs2 u:object_r:labeledfs:s0;
```
<p style="text-indent:2em">
对于虚拟文件系统，即 Linux 运行过程中创建的 VFS ，则使用 fs_use_task 关键字描述，目前仅有 pipefs 和 socketfs 两种 VFS 格式，例如：
</p>

```
fs_use_task pipefs u:object_r:pipefs:s0;
fs_use_task sockfs u:object_r:sockfs:s0;
```

<p style="text-indent:2em">
还有一个 fs_use_trans ，它也是用于 VFS ，但根据 SELinux 官方描述，这些 VFS 是针对 pesudo terminal 和临时对象。在具体 labeling 时，会根据 fs_use_trans 以及 Type Transition 的规则来决定最终的 SContext 。例如：
</p>

```
fs_use_trans devpts u:object_r:devpts:s0;
fs_use_trans tmpfs u:object_r:tmpfs:s0;
fs_use_trans devtmpfs u:object_r:device:s0;
fs_use_trans shm u:object_r:shm:s0;
fs_use_trans mqueue u:object_r:mqueue:s0;
```
<p style="text-indent:2em">
假设有下面一条策略，表示当 sysadm_t 的进程在 Type 为 devpts 下创建一个 chr_file 时，其 SContext 将是 sysadm_devpts_t:s0 。如果没有这一条 TE ，则将使用 fs_use_trans 设置的 SContext 为 u:object_r:devpts:s0 。
</p>

```
type_transition sysadm_t devpts:chr_file sysadm_devpts_t:s0;
```
#### 2.5.4.3 genfs_context 文件
<p style="text-indent:2em">
external/sepolicy/genfs_context 文件，一般就是根目录、proc目录、sysfs目录，需要使用 genfscon 关键字来打 labeling 。
</p>

```
genfscon rootfs / u:object_r:rootfs:s0
genfscon proc / u:object_r:proc:s0
genfscon proc /net/xt_qtaguid/ctrl u:object_r:qtaguid_proc:s0
```

## 2.6.	Security Level 和 MLS
### 2.6.1. Security Level
<p style="text-indent:2em">
SELinux 中添加了多等级安全管理，即 Multi-Level Security 。多等级安全信息也被添加到 SContext 中。所以，在 MLS 启用的情况下，完整的 SContext 规范如下：
</p>

```
user:role:type:sensitivity[:category,...]- sensitivity [:category,...]
```
<p style="text-indent:2em">
上面 sensitivity[:category,...]- sensitivity [:category,...] 表示 Security-level 范围 [low security level] -  [high security level] ，由三部分组成：
</p>

* low security level ：表明当前 SContext 所对应的当前(也就是最小)安全级别
* 连字符 “-” ，表示 range 
* high security level ：表明当前 SContext 所对应最高可能获得的安全级别

security level 由两部分组成，先来看第一部分由 sensitivity 关键字定义的 sensitivity ，用 sensitivity 定义一个 sens_id ， alias 指定别名，格式如下：

```
sensitivity sens_id alias alias_id [ alias_id ]; 
sensitivity s0 alias unclassified
sensitivity s1 alias seceret
sensitivity s2 alias top-seceret
```
<p style="text-indent:2em">
alias 并不是 sensitivity 的必要选项，且名字可以任取。在 SELinux 中，真正设置 sensitivity 级别的是由下面这个关键词表示： dominance {s0 s1 s2.....sn} ，在 dominance 语句中，括号内最左边的 s0 级别最低，依次递增，直到最右边的 sn 级别最高。
</p>

<p style="text-indent:2em">
再来看 security level 第二部分，即 category 关键字及用法。 Category 语法规范为： category cat_id alias alias_id; 例如：
</p>

```
category c0
category c1
```
<p style="text-indent:2em">
category 和 sensitivity 不同，它定义的是类别，类别之间是没有层级关系的。 SEAndroid 中， sensitivity 只定义了 s0 ， category 定义了 c0-c1023 。 senstivity 和 category 一起组成了一个 security level(以后简称SLevel)， SLevel 由关键字level 声明，如下例所示： level sens_id [ :category_id ]; 
</p>
<p style="text-indent:2em">
举例： level s0:c0.c255;  sensitivity 为 s0 ， category 从 c0,c1,c2 一直到 c255 ，注意其中的 . 号。 SLevel 可以没有 category_id 。看一个例子： level s0 。和 Role 类似， SL1 和 SL2 之间的关系有：
</p>

* dom ：如果 SL1 dom SL2 的话，则 SL1 的 sensitivity >= SL2 的 senstivity ， SL1 的 category 包含 SL2 的 category (即 Category of SL1 是 Category of SL2 的超集)。例如： SL1="s2:c0.c5" dom SL2="s0:c2.c3"
* domby ：和 dom 相反
* eq ： sensitivity 相等， category 相同
* incomp ：不可比。 sensitivity 不可比， category 也不可比

<p style="text-indent:2em">
现在回过头来看SContext，其完整格式为：
</p>

```
user:role:type:sensitivity[:category,...]- sensitivity [:category,...]
```

<p style="text-indent:2em">
前面例子中，我们看到 Android 中， SContext 有： u:r:init:s0，在这种情况下， Low SLevel 等于 High SLevel ，而且 SLevel 没有包含 Category 。
</p>

### 2.6.2. mlsconstrain
<p style="text-indent:2em">
了解了 SLevel 后，下面来看看它如何在 MAC 中发挥自己的力量。和 constrain 类似， MLS 在其基础上添加了一个功能更强大的 mlsconstrain 关键字。 mlsconstrain 语法和 constrain 一样： 
</p>

```
mlsconstrain class perm_set expression;
```
<p style="text-indent:2em">
和 constrain 不一样的是， expression 除了 u1,u2,r1,r2,t1,t2 外还新增了： l1,l2 ， l1 表示 source 的 low senstivity level 。 l2 表示 target 的 low sensitivity ； h1,h2 ： h1 表示 source 的 high senstivity level 。 h2 表示 target 的 high sensitivity 。 l 和 h 的关系，包括 dom,domby,eq 和 incomp 。 mlsconstrain 只是一个 Policy 语法，那么我们应该如何充分利用它来体现多层级安全管理呢？看下图：
</p>

![MLS 图例](/img/selevel.png)

<p style="text-indent:2em">
MLS在安全策略上有一个形象的描述叫 no write down 和 no read up ：
</p>

* 高级别的东西不能往低级别的东西里边写数据：这样可能导致高级别的数据泄露到低级别中。如下图中， Process 的级别是 Confidential ，它可以往同级别的 File B 中读写数据，但是只能往高级别的 File A(级别是 Secret)里边写东西
* 高级别的东西只能从低级别的东西里边读数据：比如 Process 可以从 File C 和 File D 中读数据，但是不能往 File C 和 File D 上写数据

### 2.6.3. MLS in SEAndroid
<p style="text-indent:2em">
再来看看 SEAndroid 中的 MLS ：系统中只有一个 sensitivity level ，即 s0 。系统中有 1024 个 category ，从 c0 到 c1023 。
</p>

# 3. 编译安全策略文件
<p style="text-indent:2em">
在 Android 源码的 external/sepolicy/ 目录下有各种文件，这些文件会被组合到一起，组合到一起后的文件中有使用宏的地方，需要利用 m4 命令对这些宏进行扩展。 m4 命令处理完成后得到的文件叫 policy.conf ，它是所有安全策略源文件的集合，宏也被替换了。所以，我们可以通过该文件查看整个系统的安全配置情况。 policy.conf 文件最终要被 checkpolicy 命令处理，该命令要检查 neverallow 是否被违背，语法是否正确等。最后， checkpolicy 会将 policy.conf 打包成二进制文件。在 SEAndroid 中，该文件叫 sepolicy 。最后，我们再把这个 sepolicy 文件传递到 kernel LSM 中，整个安全策略配置就算完成了。
</p>






