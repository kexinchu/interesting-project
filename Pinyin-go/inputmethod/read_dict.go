package inputmethod

import (
	"bufio"
	"errors"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
)

// readLocalFile 读本地文件
func readLocalFile(filePath, spell string) (wordInfos, error) {
	// 读取词典文件
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return dataProcess(file, spell)
}

// readRemoteFile 读远程文件
func readRemoteFile(httpPath, spell string) (wordInfos, error) {
	// get 远程数据
	resp, err := http.Get(httpPath)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return dataProcess(resp.Body, spell)
}

// dataProcess 数据处理
func dataProcess(file io.Reader, pinyin string) (wordInfos, error) {
	// 存储分数-汉字
	var wordList wordInfos
	rd := bufio.NewReader(file)
	for {
		// 逐行
		line, _, err := rd.ReadLine()
		if err != nil {
			break
		}
		strLine := string(line)
		// 字典里的汉字与分数以空格隔开
		wordAndScore := strings.Split(strLine, " ")
		if len(wordAndScore) != 2 {
			continue
		}
		// 字符串形式的分数转换为整数
		score, err := strconv.Atoi(wordAndScore[1])
		if err != nil || score <= 0 {
			continue
		}
		// 写word
		wordList = append(wordList, wordInfo{
			spell: pinyin,
			word:  wordAndScore[0],
			frac:  score,
		})
	}
	// 判空
	if len(wordList) == 0 {
		return nil, errors.New("empty file")
	}
	return wordList, nil
}
