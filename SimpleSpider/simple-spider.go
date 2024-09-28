package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"golang.org/x/net/html"
)

const COOKIE = "cookie"

func main() {

	link := "111"

	client := &http.Client{
		Transport: &http.Transport{
			Dial: (&net.Dialer{
				Timeout: time.Duration(1) * time.Second,
			}).Dial,
		},
		Timeout: time.Duration(1) * time.Second,
	}
	req, err := http.NewRequest("GET", link, nil)
	req.Header.Set("Cookie", COOKIE)

	res, err := client.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fetch: %v\n", err)
		os.Exit(1)
	}

	// // 读取资源数据 body: []byte
	all, err := ioutil.ReadAll(res.Body)
	// 关闭资源流
	res.Body.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fetch: reading %s: %v\n", link, err)
		os.Exit(1)
	}

	doc, _ := html.Parse(strings.NewReader(string(all)))
	bodys, err := getValue(doc, "body")
	if err != nil || len(bodys) != 1 {
		return
	}

	var value map[string]interface{}
	for _, body := range bodys {
		codes, err := getValue(body, "code")
		if err != nil || len(codes) != 1 {
			return
		}
		code := renderNode(codes[0])

		value1 := extractAbs(code.Bytes())

		err = json.Unmarshal(value1, &value)
		if err != nil {
			fmt.Println("Unmarshal failed")
		}
		fmt.Println(value)
		// fmt.Printf(string(code.String()))
	}

	// newBogy := renderNode(bn)

	// 控制台打印内容 以下两种方法等同
	// fmt.Printf("%s", body)
}

// 提取html中的key值
// s := `<p>Links:</p><ul><li><a href="foo">Foo</a><li><a href="/bar/baz">BarBaz</a></ul>`
// doc, err := html.Parse(strings.NewReader(s))
// key = "a"
// 返回:
//   <a href="foo">Foo</a><li>
//   <a href="/bar/baz">BarBaz</a>
func getValue(doc *html.Node, key string) ([]*html.Node, error) {
	var b []*html.Node
	var f func(*html.Node)
	f = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == key {
			b = append(b, n)
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			f(c)
		}
	}
	f(doc)
	if len(b) > 0 {
		return b, nil
	}
	return nil, errors.New("Missing <" + key + "> in the node tree")
}

func renderNode(n *html.Node) bytes.Buffer {
	var buf bytes.Buffer
	w := io.Writer(&buf)
	html.Render(w, n)
	return buf
}

func extractAbs(value []byte) []byte {
	// string
	strValue := string(value)
	strValue = strings.Replace(strValue, "<code style=\"white-space: pre-line\">", "", -1)
	strValue = strings.Replace(strValue, "</code>", "", -1)
	strValue = strings.Replace(strValue, "&#34;", "\"", -1)
	strValue = strings.Replace(strValue, "\n", "", -1)
	return []byte(strValue)
}
