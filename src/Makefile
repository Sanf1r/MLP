FLAGS=-Wall -Wextra -Werror -std=c++17 -pedantic
GTEST=-lgtest -lgtest_main -pthread
COVER=-fprofile-arcs -ftest-coverage
ASAN=#-fsanitize=address -fanalyzer
FILE_NAME_S=MLP
FILE_NAME_L=MLP-1.0

.PHONY: all
all: install start

.PHONY: install
install:
	cmake -S . -B ./build && cd ./build && make

start:
	./build/$(FILE_NAME_S)

.PHONY: uninstall
uninstall:
	rm -rf build

.PHONY: open
open:
	cd build && ./$(FILE_NAME_S)

.PHONY: clean
clean:
	rm -rf build
	rm -rf $(FILE_NAME_L)
	rm -f $(FILE_NAME_L).tar.gz
	rm -f *.info *.gcda *.gcno *.gcov *.css *.html
	rm -rf coverage
	rm -rf test

.PHONY: dvi
dvi2: 
	makeinfo --html $(FILE_NAME_L).texi
	open $(FILE_NAME_L)/index.html 

.PHONY: dist
dist:
	tar czvf $(FILE_NAME_L).tar.gz ./Model ./View ./Controller \
	CMakeLists.txt main.cc Makefile $(FILE_NAME_L).texi

.PHONY: tests
tests: 
	rm -f *.gcda *.gcno
	g++ $(FLAGS) $(ASAN) $(GTEST) $(COVER) \
	./Testing/tests.cc ./Model/Model.cc ./Model/MatrixNN.cc ./Model/GraphNN.cc ./Model/DataParse.cc -o test
	./test
	geninfo . -b . -o ./coverage.info --no-external
	genhtml coverage.info -o ./coverage

.PHONY: gcov_report
gcov_report:
	open coverage/index.html

lint:
	clang-format -n *.cc ./Model/*.cc ./View/*.cc
	clang-format -n ./Model/*.h ./View/*.h ./Controller/*.h

leaks:
	leaks -atExit -- ./test

