#!/bin/bash
#脚本以 file_size.sh [-n N] [-d DIR]的格式运行，当不给脚本任何参数时，默认输出当前路径最大的前10个文件。当按格式给予参数时，将输出路径DIR所指的文件夹中最大的前N个文件
function file_size()   #将指定文件夹内文件按大小排序，输出最大的前N个文件
{
    if [ ! -n "$2" ] ;then  #根据是否给予参数来决定指定文件夹的路径
	dir=`pwd`
    else
	dir=$2
    fi
    du -ak --max-depth=1 $dir | sort -nrk 1 > temp.txt  #将文件按大小排序，结果存入temp.txt中
    cat -n temp.txt > temp1.txt   #排序结果进行编号
    head -n $1 temp1.txt          #输出最大的前N位结果
    rm temp.txt                   #删除temp.txt与temp1.txt文件
    rm temp1.txt
}

#根据传参结果判断是否有传参、传参格式是否正确，然后给予不同的输出
if [ ! -n "$1" ] && [ ! -n "$2" ] && [ ! -n "$3" ] && [ ! -n "$4" ] ;then
	file_size 10 $4
elif [[ $1 != '-n' ]]||[[ $3 != '-d' ]] ;then
    echo -e "usage: file_size.sh [-n N] [-d DIR] \nShow top N largest files/directories"
else
    file_size $2 $4
    
fi
