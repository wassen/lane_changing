script_dir=$(cd $(dirname $0);pwd)

url=http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz

libsvm_dir=$script_dir/lib/libsvm
mkdir -p $libsvm_dir
wget -O - $url | tar zx -C $libsvm_dir --strip-components 1
cd $libsvm_dir && make

# wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/fselect/fselect.py -P $libsvm_dir/tools
# chmod u+x $libsvm_dir/tools/fselect.py

ln -s `pwd`/fselect.py `pwd`/lib/libsvm/tools/fselect.py
