#!/bin/bash
#脚本支持以 task2.sh [Source compressed file] [Destination path]的格式对目标文件进行解压缩，并且支持task3.sh --list的形式查询脚本可解压缩的文件类型

function self_compression()        #获取目标文件的后缀名，根据文件格式进行解压缩
{
    n=$1
    extension=${n##*.}
    if [[ $extension == 'zip' ]] ;then
	unzip $1 -d $2;
    elif [[ $extension == 'tar' ]] ;then
	tar -xvf $1 -C $2;
    elif [[ $extension == 'gz' ]] ;then
	tar -xzvf $1 -C $2;
    elif [[ $extension == 'bz2' ]] ;then
	tar -xjvf $1 -C $2;
    else
	echo -e "error\nSupported file types: zip tar tar.gz tar.bz2"
    fi
        
}

if [ ! -n "$1" ] ;then                #无传参是输出usage
    echo -e "usage: task2.sh [--list] or [Source compressed file] [Destination path] \nSelf compression according to the file name suffix"
elif [[ $1 == '--list' ]] ;then       #传参为--list时，输出支持的压缩文件的格式
    echo "Supported file types: zip tar tar.gz tar.bz2"
else
    self_compression $1 $2
fi
