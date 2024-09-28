package inputmethod

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestInputMethod_NewInputMethod(tt *testing.T) {
	tests := []struct {
		name      string
		filepaths []string
		want      *SpellInputMethod
	}{
		{
			name:      "test1",
			filepaths: []string{"../testdata/dong.dat", "../testdata/nan.dat", "../testdata/xi.dat"},
			want:      NewInputMethod([]string{"../testdata/dong.dat", "../testdata/nan.dat", "../testdata/xi.dat"}),
		},
	}

	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			get := NewInputMethod(test.filepaths)
			if !reflect.DeepEqual(get, test.want) {
				tt.Errorf("NewInputMethod() get %v, want %v", get, test.want)
			}
		})
	}
}

func TestInputMethod_readDataFile(tt *testing.T) {
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
			filepath: url + "/dong.dat",
			spell:    "dong",
			want:     wordInfos{{spell: "dong", word: "东", frac: 9}},
		},
	}

	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			if get, _ := readDataFile(test.filepath, test.spell); !reflect.DeepEqual(get, test.want) {
				tt.Errorf("readDataFile() get %v, want %v", get, test.want)
			}
		})
	}
}

func ConstructInputMethod() *SpellInputMethod {
	/**
	 * 读取测试文件，构建SpellInputMethod; 方便测试
	 */

	test := struct {
		filepaths []string
	}{
		filepaths: []string{"../testdata/dong.dat", "../testdata/nan.dat", "../testdata/xi.dat"},
	}

	return NewInputMethod(test.filepaths)
}

func TestInputMethod_FindWords(tt *testing.T) {
	// 构建用于测试的Trie
	sim := ConstructInputMethod()

	tests := []struct {
		name  string
		spell string
		want  []string
	}{
		{
			name:  "test1",
			spell: "dong",
			want:  []string{"东", "洞", "懂", "动"},
		},
		{
			name:  "test2",
			spell: "nan",
			want:  []string{"南", "男", "难", "楠"},
		},
		{
			name:  "test3",
			spell: "xi",
			want:  nil,
		},
		{
			name:  "test4",
			spell: "bei",
			want:  nil,
		},
		{
			name:  "test4",
			spell: "",
			want:  nil,
		},
	}
	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			if get := sim.FindWords(test.spell); !reflect.DeepEqual(get, test.want) {
				tt.Errorf("SpellInputMethod.FindWords() = %v, want %v", get, test.want)
			}
		})
	}
}
