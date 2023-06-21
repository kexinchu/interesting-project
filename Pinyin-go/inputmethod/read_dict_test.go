package inputmethod

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"
)

func TestReadDict_readLocalFileTest(tt *testing.T) {
	tests := []struct {
		name     string
		filepath string
		spell    string
		want     wordInfos
	}{
		{
			name:     "test1",
			filepath: "../testdata/dong.dat",
			spell:    "dong",
			want: wordInfos{
				{spell: "dong", word: "东", frac: 9},
				{spell: "dong", word: "懂", frac: 8},
				{spell: "dong", word: "动", frac: 7},
				{spell: "dong", word: "洞", frac: 9},
			},
		},
		{
			name:     "test2",
			filepath: "../testdata/nan.dat",
			spell:    "nan",
			want: wordInfos{
				{spell: "nan", word: "南", frac: 9},
				{spell: "nan", word: "男", frac: 9},
				{spell: "nan", word: "难", frac: 8},
				{spell: "nan", word: "楠", frac: 7},
			},
		},
		{
			name:     "test3",
			filepath: "../testdata/xi.dat",
			spell:    "xi",
			want:     nil,
		},
		{
			name:     "test4",
			filepath: "../testdata/bei.dat",
			spell:    "bei",
			want:     nil,
		},
	}

	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			words, _ := readLocalFile(test.filepath, test.spell)
			if !reflect.DeepEqual(words, test.want) {
				tt.Errorf("readLocalFile() get : %v want : %v", words, test.want)
			}
		})
	}
}

func TestReadDict_readRemoteFileTest(tt *testing.T) {
	// 使用httptest 构建测试
	testHTTP := httptest.NewServer(
		http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				// 获取url链接最后的文件名
				if r.URL.EscapedPath() == "/dong.dat" {
					fmt.Fprintln(w, "东 9")
				}
			},
		),
	)
	defer testHTTP.Close()
	url := testHTTP.URL // 测试链接前缀

	tests := []struct {
		name     string
		httppath string
		spell    string
		want     wordInfos
	}{
		{
			name:     "test1",
			httppath: url + "/dong.dat",
			spell:    "dong",
			want:     wordInfos{{spell: "dong", word: "东", frac: 9}},
		},
		{
			name:     "test2",
			httppath: url + "/nan.dat",
			spell:    "nan",
			want:     nil,
		},
		{
			name:     "test3",
			httppath: "https://",
			spell:    "",
			want:     nil,
		},
	}

	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			words, _ := readRemoteFile(test.httppath, test.spell)
			if !reflect.DeepEqual(words, test.want) {
				tt.Errorf("readRemoteFile() get : %v want : %v", words, test.want)
			}
		})
	}
}

func TestReadDict_dataProcessTest(tt *testing.T) {
	tests := []struct {
		name     string
		filepath string
		spell    string
		want     wordInfos
	}{
		{
			name:     "test1",
			filepath: "../testdata/dong.dat",
			spell:    "dong",
			want: wordInfos{
				{spell: "dong", word: "东", frac: 9},
				{spell: "dong", word: "懂", frac: 8},
				{spell: "dong", word: "动", frac: 7},
				{spell: "dong", word: "洞", frac: 9},
			},
		},
		{
			name:     "test2",
			filepath: "../testdata/nan.dat",
			spell:    "nan",
			want: wordInfos{
				{spell: "nan", word: "南", frac: 9},
				{spell: "nan", word: "男", frac: 9},
				{spell: "nan", word: "难", frac: 8},
				{spell: "nan", word: "楠", frac: 7},
			},
		},
	}

	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			// 打开本地文件
			file, _ := os.Open(test.filepath)
			defer file.Close()
			// 测试
			words, _ := dataProcess(file, test.spell)
			if !reflect.DeepEqual(words, test.want) {
				tt.Errorf("readLocalFile() get : %v want : %v", words, test.want)
			}
		})
	}
}
