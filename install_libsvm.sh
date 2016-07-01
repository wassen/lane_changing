script_dir=$(cd $(dirname $0);pwd)

url=http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz

libsvm_dir=$script_dir/lib/libsvm
mkdir -p $libsvm_dir
wget -O - $url | tar zx -C $libsvm_dir --strip-components 1
cd $libsvm_dir && make
