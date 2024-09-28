package inputmethod

import (
	"path/filepath"
	"strings"
)

// SpellInputMethod 为输入法实例
type SpellInputMethod struct {
	// 字典树用于快速查询某个拼音是否存在，存在则可以调用FindWords方法返回该结点存储的汉字
	TrieTree Trie
}

// NewInputMethod 根据词典文件list，创建实例
func NewInputMethod(filePaths []string) *SpellInputMethod {
	inputMethod := &SpellInputMethod{
		TrieTree: Trie{},
	}

	for i := range filePaths {
		oneFile := filePaths[i]

		// 获取文件名
		fileName := filepath.Base(oneFile)
		// 获取拼音
		spellArr := strings.Split(fileName, ".")
		if len(spellArr) != 2 {
			continue
		}
		spell := spellArr[0]

		// 从文件中读取汉字 + 分数 并存储到 Trie 树中
		wordList, err := readDataFile(oneFile, spell)
		if err != nil {
			continue
		}
		inputMethod.TrieTree.insert(spell, wordList)
	}
	return inputMethod
}

// readDataFile 读取词典文件，将 汉字-分数 存储到map中
func readDataFile(filePath, spell string) (wordInfos, error) {
	// 存储分数-汉字
	if strings.HasPrefix(filePath, "http://") || strings.HasPrefix(filePath, "https://") {
		// 远程文件
		return readRemoteFile(filePath, spell)
	}
	// 本地文件
	return readLocalFile(filePath, spell)
}

// FindWords 根据输入的拼音返回对应的中文汉字
func (sim *SpellInputMethod) FindWords(spell string) (wordList []string) {
	// 无效spell
	if len(spell) == 0 {
		return nil
	}
	// 在 Trie 树中查找 spell 对应的 node
	node, ok := sim.TrieTree.search(spell)
	if !ok {
		return nil
	}
	// 根据 node 获取 words
	wordList = getWords(node)
	return wordList
}
