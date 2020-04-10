---
layout:     post
title:      Linux PAM机制
#subtitle:  
date:       2019-1-19
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - Linux
    - PAM
---

> [我的博客](http://feizaipp.github.io)

<p style="text-indent:2em">
Linux操作系统最大的优点之一就是支持多用户，Linux系统通过/etc/passwd和/etc/shadow两个文件管理系统的所有用户名和密码。在用户登录Linux系统时，Linux系统提供了一套统一的验证用户身份的接口，即PAM。Linux-PAM(Pluggable Authentication Modules for Linux)，即Linux可插入认证模块，它是一套共享库，放在/lib64/security下。系统管理员通过修改/etc/pam.d/下的配置文件，来管理对程序的认证方式。</p>
<p style="text-indent:2em">
应用程序调用相应的配置文件，从而调用本地的认证模块。像我们使用su命令时，系统会提示你输入root用户的密码，PAM模块读取密码后，对密码进行校验，读取密码和校验密码的正确性就是su程序通过调用PAM模块实现的。</p>

# 1.PAM的配置文件说明
要使用PAM模块的服务必须在/etc/pam.d/目录下面实现以服务名为文件名的配置文件，比如su程序要使用PAM模块，就要实现/etc/pam.d/su文件，内容如下：

> \#%PAM-1.0  
> auth            sufficient      pam_rootok.so  
> \# Uncomment the following line to implicitly trust users in the "wheel" group.  
> \#auth           sufficient      pam_wheel.so trust use_uid  
> \# Uncomment the following line to require a user to be in the "wheel" group.  
> \#auth           required        pam_wheel.so use_uid  
> auth            substack        system-auth  
> auth            include         postlogin  
> account         sufficient      pam_succeed_if.so uid = 0 use_uid quiet  
> account         include         system-auth  
> password        include         system-auth  
> session         include         system-auth  
> session         include         postlogin  
> session         optional        pam_xauth.so  

由上面的pam模块文件内容看，可以将pam配置文件分为四列，第一列代表模块类型；第二列代表控制模式；第三列代表模块路径；第四列代表模块参数。

# 1.1.模块类型
Linux-PAM有四种模块类型，分别代表四种不同的任务，它们分别是，认证管理（auth），账号管理（account），会话管理（session）和密码（password）管理，一个类型可能有多行，它们按顺序依次由PAM模块调用。

管理方式 | 说明
-|-
auth | 用来对用户身份进行识别。如提示用户输入密码，或判断用户是否为root等。
account | 对账户的各项属性进行检查。如是否允许登录，是否达到最大用户数，或是root用户是否允许在这个终端登录。
session | 这个模块用来定义用户登录前，以及用户退出后所要进行的操作。如登录连接信息，用户数据的打开与关闭。挂载文件系统等。
password | 更新用户信息。如修改用户密码。

# 1.2.控制模式
用于定义各个认证模块在给出各种结果时 PAM 的行为，或者调用在别的配置文件中定义的认证流程栈。该列有两种形式，一种是比较常见的“关键字”模式，另一种则是用方括号([])包含的“返回值=行为”模式。

# 1.2.1.关键字模式

控制标记 | 说明
-|-
required | 如果本条目没有被满足，那最终本次认证一定失败，但认证过程不因此打断。整个栈运行完毕之后才会返回（已经注定了的）“认证失败”信号。
requisite | 如果本条目没有被满足，那本次认证一定失败，而且整个栈立即中止并返回错误信号。
sufficient | 如果本条目的条件被满足，且本条目之前没有任何required条目失败，则立即返回“认证成功”信号；如果对本条目的验证失败，不对结果造成影响。
optional | 表示即使本行指定的模块验证失败，也允许用户接受应用程序提供的服务，一般返回PAM_IGNORE(忽略)。
include | 表示在验证过程中调用其他的PAM配置文件。在CentOS 7系统中有相当多的应用通过调用/etc/pam.d/system-auth来实现认证而不需要重新逐一去写配置项。
substack | 运行其他配置文件中的流程，并将整个运行结果作为该行的结果进行输出。该模式和 include 的不同点在于认证结果的作用域：如果某个流程栈 include 了一个带 requisite 的栈，这个 requisite 失败将直接导致认证失败，同时退出栈；而某个流程栈 substack 了同样的栈时，requisite 的失败只会导致这个子栈返回失败信号，母栈并不会在此退出。

# 1.2.2. 返回值=行为模式
这种模式更为复杂。格式如下：

[value1 = action1 value2 = action2 ……]

其中，valueN 的值是各个认证模块执行之后的返回值。有 success、user_unknown、new_authtok_reqd、default 等等数十种。其中，default 代表其他所有没有明确说明的返回值。返回值结果清单可以在 /usr/include/security/_pam_types.h 中找到，也可以查询 pam(3) 获取详细描述。

流程栈中很可能有多个验证规则，每条验证的返回值可能不尽相同，那么到底哪一个验证规则能作为最终的结果呢？这就需要 actionN 的值来决定了。actionN 的值有以下几种：

ignore：在一个栈中有多个认证条目的情况下，如果标记 ignore 的返回值被命中，那么这条返回值不会对最终的认证结果产生影响。

bad：标记 bad 的返回值被命中时，最终的认证结果注定会失败。此外，如果这条 bad 的返回值是整个栈的第一个失败项，那么整个栈的返回值一定是这个返回值，后面的认证无论结果怎样都改变不了现状了。

die：标记 die 的返回值被命中时，马上退出栈并宣告失败。整个返回值为这个 die 的返回值。

ok：在一个栈的运行过程中，如果 ok 前面没有返回值，或者前面的返回值为 PAM_SUCCESS，那么这个标记了 ok 的返回值将覆盖前面的返回值。但如果前面执行过的验证中有最终将导致失败的返回值，那 ok 标记的值将不会起作用。

done：在前面没有 bad 值被命中的情况下，done 值被命中之后将马上被返回，并退出整个栈。

N（一个自然数）：功效和 ok 类似，并且会跳过接下来的 N 个验证步骤。如果 N = 0 则和 ok 完全相同。

reset：清空之前生效的返回值，并且从下面的验证起重新开始。

我们在前文中已经介绍了控制模式(control)的“关键字”模式。实际上，“关键字”模式可以等效地用“返回值=行为”模式来表示。具体的对应如下：
> required:  
> [success=ok new_authtok_reqd=ok ignore=ignore default=bad]  
> requisite:  
> [success=ok new_authtok_reqd=ok ignore=ignore default=die]  
> sufficient:  
> [success=done new_authtok_reqd=done default=ignore]  
> optional:  
> [success=ok new_authtok_reqd=ok default=ignore]  

# 1.3.模块路径
模块路径，即要调用模块的位置。如果是64位系统，一般保存在/lib64/security，同一个模块，可以出现在不同的类型中，它在不同的类型中所执行的操作都不相同。这是由于每个模块，针对不同的模块类型，编制了不同的执行函数。

# 1.4.模块参数
模块参数，即传递给模块的参数。参数可以有多个，之间用空格分隔开，如：password required pam_unix.so nullok obscure min=4 max=8 md5。

# 2.PAM模块工作流程
PAM模块提供多个动态库，供其他应用程序使用。通过su程序使用PAM模块的流程，分析PAM模块的工作流程。首先，初始化PAM环境，解析配置文件，根据配置文件内容初始化handler。如下图所示：  
![PAM模块初始化](/img/su流程图.png)

1）su程序在shadow包里，我们在执行su命令切换到root账户，需要输入root账户的密码，这里就需要对用户输入的root密码进行验证，验证的过程是PAM模块来完成的。pam_start函数分配一个pam_handle_t结构体，并在后续的处理中对该结构体进行初始化。

2）_pam_init_handlers函数主要解析配置文件，设置pam_handle结构体中的service结构体，service结构体用来记录配置文件对应的处理函数。

3）_pam_parse_conf_file函数逐行解析配置文件，对于su程序来说解析/etc/pam.d/su文件，#号开头行忽略。

4）在_pam_parse_conf_file函数中解析控制模式，首先解析“关键字”模式，如果不是关键字模式就解析“返回值=行为”模式，这两种模式上面的1.2控制模式中介绍过了。

5）_pam_add_handler函数将配置文件的每一行的模块名动态加载到本地，然后根据模块类型加载模块中的函数，并将函数放到handler结构体中。

6）_pam_load_module动态加载模块。

7）_pam_dlopen加载模块，并获得句柄。

8）返回loaded_module结构体。

9）初始化handler结构体。

10）将pam_handle_t返还给使用者。

11）check_perms函数使用PAM模块。

然后，调用handler中的处理函数，进行身份鉴别，最终结果为PAM_SUCCESS则认证成功。如下图所示：  
![使用PAM模块](/img/su流程图1.png)