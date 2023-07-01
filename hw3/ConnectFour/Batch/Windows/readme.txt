1. 新建“Sourcecode”、“TempResult”、“dll”、“compete_result”目录。
	1）“Sourcecode”目录下以自己的学号建立子目录，存放strategy项目源代码文件,包括Judge.cpp，Judge.h，Strategy.cpp，Strategy.h，Point.h以及自定义的*.cpp和*.h文件，不要包含其他源码文件。
	2）“dll”目录下存放自己的dll文件，以学号命名。
	3）“TestCases”目录下存放测试样例（注意：所有测试样例需以两位数编号，例如02.dll、04.dll、06.dll、	08.dll，100.dll以00.dll表示）。
2. compile.bat:
	1) 将strategy项目下的源代码编译为可执行dll文件。
	2) 需修改第2行call之后的路径为自己Visual Studio安装路径。
3. compete.bat: 对抗测试
4. stat.py: 统计得分
5. 运行顺序：compile.bat—>compete.bat—>stat.py