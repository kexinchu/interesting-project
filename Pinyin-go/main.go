
// Package main is special.  It defines a
// standalone executable program, not a library.
// Within package main the function main is also
// special—it’s where execution of the program begins.
// Whatever main does is what the program does.

package main

import (
	"bufio"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"

	"github.com/kexinchu/interesting-project/Pinyin-go/inputmethod"
)

// loop 循环读取用户输入的spell, 查询中文
func loop(im *inputmethod.SpellInputMethod) {
	// 从标准输入流中接收输入数据
	stdin := bufio.NewReader(os.Stdin)
	for {
		spell, err := stdin.ReadString('\n')
		if !errors.Is(err, nil) {
			break
		}
		spell = strings.TrimRight(spell, "\n")
		words := im.FindWords(spell)
		fmt.Println(strings.Join(words, ", "))
	}
}

// addFilePaths 加载本地测试文件 - 根据文件夹
func addFilePaths(dir string) []string {
	fileInfo, err := ioutil.ReadDir(dir)
	if !errors.Is(err, nil) {
		log.Println("fail to open dir ", dir)
		panic(err)
	}
	// make([]type, len, cap)
	filePaths := make([]string, 0, len(fileInfo))
	for i := range fileInfo {
		// 非 .dat 文件
		if !strings.HasSuffix(fileInfo[i].Name(), ".dat") {
			continue
		}
		filePaths = append(filePaths, dir+fileInfo[i].Name())
	}
	if len(filePaths) == 0 {
		log.Println("there is no dict file in ", dir)
		panic(err)
	}
	return filePaths
}

// main the function where execution of the program begins
func main() {
	filePaths := os.Args[1:]
	if len(filePaths) == 0 {
		filePaths = addFilePaths("./dict/")
	}
	im := inputmethod.NewInputMethod(filePaths)
	loop(im)
}
