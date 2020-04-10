---
layout:     post
title:      自动生成Makefile
#subtitle:  
date:       2019-1-30
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - Linux
    - Makefile
    - autoconf
    - automake
---

> [我的博客](http://feizaipp.github.io)

<p style="text-indent:2em">
对于 Linux 的初学者，搭建开发环境的一直是绕不过去的坎，其中最难得当属编译环境的配置了，复杂的 Makefile 语法让很多初学者望而却步。遗憾的是，不管你是自己进行项目开发还是安装应用软件，我们都经常要用到 make 、 make install 等工具。因为利用 make 工具，我们可以将大型的开发项目分解成为多个更易于管理的模块，对于一个包括几百个源文件的应用程序，使用 make 和 Makefile 工具就可以轻而易举的理顺各个源文件之间纷繁复杂的相互关系。幸而有 GNU 提供的 autoconf 及 automake 这两套工具使得编写 Makefile 不再是一个难题。</p>
<p style="text-indent:2em">
本文将介绍如何利用 GNU autoconf 及 automake 这两套工具来协助我们自动产生 Makefile 文件，并且让开发出来的软件可以像大多数源码包那样，只需"./configure", "make","make install" 就可以把程序安装到系统中。
</p>

# 1.自动生成 Makefile 的一般步骤
* 运行 autoscan 命令  
该命令扫描源代码以搜寻普通的可移植性问题，比如检查编译器、库、头文件等，生成文件configure.scan，它是configure.ac的一个雏形。
* 将 configure.scan 文件重命名为 configure.ac ，并修改  
下面通过一个例子介绍 configuer.ac 的语法。
* 执行aclocal命令  
aclocal 是一个 perl 脚本程序。 aclocal 根据 configure.ac 文件的内容，自动生成 aclocal.m4 文件。
* 执行autoheader命令  
该命令生成 config.h.in 文件。该命令通常会从 acconfig.h 文件中复制用户附加的符号定义。
* 执行 autoconf 命令  
有了 configure.ac 和 aclocal.m4 两个文件以后，我们就可以使用 autoconf 来产生 configure 文件了。 configure 脚本能独立于 autoconf 运行，且在运行的过程中，不需要用户的干预。
* 在项目目录及子目录下新建 Makefile.am 文件，并修改  
下面通过一个例子介绍 Makefile.am 的语法。
* 运行 automake  --add-missing 命令  
automake 会根据 Makefile.am 文件产生一些文件，其中最重要的是 Makefile.in 文件。  
注意：如果运行 automake --add-missing 后出现如下提示：  
Makefile.am: error: required file './NEWS' not found  
Makefile.am: error: required file './README' not found  
Makefile.am: error: required file './AUTHORS' not found  
Makefile.am: error: required file './ChangeLog' not found  
则要执行下面命令创建这些文件：  
touch NEWS README ChangeLog AUTHORS  
* 执行configure生成Makefile  
至此，Makefile就生成完毕了，我们可以通过make进行编译，通过make install进行安全程序了。

![生成makefile流程图](/img/automakefile.png)

# 2.应用
&#160; &#160; &#160; &#160;对于任何一样东西，最好的学习方法就是实践，其实我从一开始就没学习过 configure.ac 和 Makefile.am 的语法，当我在几个项目中使用过后我发现我已经基本掌握了这些语法。在实践的第一步就是模仿，现在大部分符合 GNU 标准的软件都用 autoconf 和 automake 工具生成Makefile 。比如我在 github 上提交的[ueventmonitor](https://github.com/feizaipp/ueventmonitor)项目。这里我们就用 ueventmonitor 项目为例，介绍 autoconf 和 automake 的使用。

## 2.1 configure.ac 语法介绍
<p style="text-indent:2em">
参照本文的第一章的内容，首先在项目的根目录执行 autoscan 命令生成 configure.scan 文件，然后将 configure.scan 文件重命名为 configure.ac 文件。首先，前两行的作用是执行 make dist 命令打包时，生成包的名字。如下所示，生成包的名字为 ueventmonitor-1.0.tar.gz 。
</p>

> AC_INIT(ueventmonitor, 1.0)  
> AC_INIT_AUTOMAKE  
<p style="text-indent:2em">
下面一行的作用是生成 config.h 文件，文件里定义了包名、版本等宏定义，在我们自己的代码里可以引用该头文件。
</p>

> AC_CONFIG_HEADERS(config.h)  

<p style="text-indent:2em">
下面这行的作用是检查用什么编译器。
</p>

> AC_PROG_CC  

<p style="text-indent:2em">
下面是重点内容，作用是检查依赖的动态库是否存在，如果存在就生成依赖。前四行判断是否存在 libdbus-glib-1.so 库，如果不存在报错，编译停止。在该库存在的情况下，最后三行用来生成依赖，生成的方法就是去 pkgconfig 目录下找 dbus-glib-1.pc 文件。将 .pc 文件的 Libs 的内容赋值给 DBUS_GLIB_LIBS 宏，将 .pc 文件的 Cflags 的内容赋值给 DBUS_GLIB_CFLAGS 宏。这些宏在 Makefile。am 文件中被使用。
</p>

> PKG_CHECK_MODULES(DBUS_GLIB, dbus-glib-1, have_dbus_glib=yes, have_dbus_glib=no)  
> if test x$have_dbus_glib = xno ; then  
>     AC_MSG_ERROR([Dbus-Glib development libraries not found])  
> fi  
> AM_CONDITIONAL(HAVE_DBUS_GLIB, test x$have_dbus_glib = xyes)  
> AC_SUBST(DBUS_GLIB_CFLAGS)  
> AC_SUBST(DBUS_GLIB_LIBS)  

<p style="text-indent:2em">
上面的内容根据需要添加，如果需要依赖其他的库，就仿照上面的例子添加即可。
</p>
<p style="text-indent:2em">
再往下是跟systemd相关的内容，如果你想让你的进程以一个系统服务运行的话需要添加这部分内容，这不是我们的重点，先忽略。
</p>

> AC_ARG_WITH([systemdsystemunitdir],  
>             AS_HELP_STRING([--with-systemdsystemunitdir=DIR], [Directory for systemd service files]),  
>             [],  
>             [with_systemdsystemunitdir=$($PKG_CONFIG --variable=systemdsystemunitdir systemd)])  
> if test "x$with_systemdsystemunitdir" != "xno"; then  
>   AC_SUBST([systemdsystemunitdir], [$with_systemdsystemunitdir])  
> fi  
> AM_CONDITIONAL(HAVE_SYSTEMD, [test -n "$systemdsystemunitdir"])  

<p style="text-indent:2em">
AC_OUTPUT 宏是我们要输出的 Makefile 的目录，也就是告诉编译器，需要编译哪些目录。
</p>

> AC_OUTPUT([Makefile  
> src/Makefile  
> data/Makefile])  
<p style="text-indent:2em">
到此为止 configure.ac 的语法就基本介绍完了，这里介绍语法是有限的，当遇到问题时再去网上找答案。
</p>

## 2.1.Makefile.am 语法介绍
### 2.1.1.根目录的 Makefile.am
<p style="text-indent:2em">
根目录的 Makefile.am 文件很简单，主要是告诉编译器，都编译哪些目录。源文件和一些默认的文件将自动打入 .tar.gz 包，其他文件若要进入 .tar.gz 包可以用 EXTRA_DIST 关键字声明，比如配置文件，数据文件等。
</p>

> SUBDIRS = data src  
> EXTRA_DIST = autogen.sh clean.sh

### 2.1.2.子目录的 Makefile.am
<p style="text-indent:2em">
子目录 src 中的 Makefile.am 文件里有两个重要的宏， AM_CPPFLAGS 和 LIBS ，这两个宏就是编译时要用的依赖，用 configure.ac 文件中定义的宏初始化，比如我们在编译目标时需要引用 dbus-glib-1 库中的接口和头文件，就需要做如下声明：
</p>

> AM_CPPFLAGS = \  
> 	$(DBUS_GLIB_CFLAGS)  

> LIBS = \  
> 	$(DBUS_GLIB_LIBS)  

<p style="text-indent:2em">
下面的几行作用是生成一个 ueventmonitord 二进制文件，安装目录时 $(libexecdir)/ueventmonitor ，需要的源文件是"main.c ueventmonitor.c ueventlinuxdevice.c"。如下所示：
</p>
ueventmonitorprivdir = $(libexecdir)/ueventmonitor
ueventmonitorpriv_PROGRAMS = ueventmonitord
ueventmonitord_SOURCES = main.c ueventmonitor.c ueventlinuxdevice.c

<p style="text-indent:2em">
如果安装目录用默认路径的话，就用 sbin_PROGRAMS 表示安装到 /usr/sbin 目录中，用 bin_PROGRAMS 表示安装到 /usr/bin 目录中。
</p>
<p style="text-indent:2em">
最后的内容是跟 dbus 相关的内容，以后在介绍。
</p>

# 3.总结
<p style="text-indent:2em">
至此，autoconfig 和 automake 相关的内容就介绍完了，当然，这短短的一篇文章不足以说明所有问题，还需要自己在项目中发现问题、解决问题。最后我将编译过程写成了脚本。在编译过程中先执行 autogen.sh 脚本，然后执行 ./configure 生成 Makefile , 最后执行 make && make install 编译并安装。自从学会了使用 autoconfig 和 automake 两个工具后，再也不用担心写 Makefile 了。
</p>