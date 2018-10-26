
util.so: util.pyx
	python cython_compile.py build_ext --inplace
