package inputmethod

import (
	"sort"
)

// DefaultAlphabetLength 默认的字母表长度大小(仅支持小写)
const DefaultAlphabetLength = 26

// Trie 字典树，
type Trie struct {
	IsWord   bool                         // 标记, 记录当前节点是否存储了一个完整拼音的Word
	Words    wordInfos                    // Words存储了对应的拼音, 汉字和分数信息
	Children [DefaultAlphabetLength]*Trie // 孩子数组
}

// insert 向 Trie 树中插入 spell，并记录分数+汉字信息
func (t *Trie) insert(spell string, words wordInfos) {
	cur := t
	for i, c := range spell {
		cIdx := c - 'a'
		// 拼音字母是否合法
		if cIdx < 0 || cIdx >= 26 {
			break
		}
		if nil == cur.Children[cIdx] {
			cur.Children[cIdx] = &Trie{}
		}
		cur = cur.Children[cIdx]
		// 判断是否完整拼音
		if len(spell)-1 == i {
			cur.IsWord = true
			cur.Words = words
		}
	}
}

// search 精确查找拼音对应的节点是否存在汉字列表
func (t *Trie) search(spell string) (*Trie, bool) {
	cur := t
	for _, c := range spell {
		cIdx := c - 'a'
		// 拼音字母是否合法
		if cIdx < 0 || cIdx >= 26 {
			return nil, false
		}
		if cur.Children[cIdx] == nil {
			return nil, false
		}
		cur = cur.Children[cIdx]
	}
	return cur, true
}

// getWords 精确查找 spell 对应的汉字列表
func getWords(node *Trie) []string {
	if node.IsWord {
		// 完整拼音，直接取该节点结果
		wordList := node.Words
		return sortAndFilter(wordList)
	}
	// 非完整拼音，去以该节点为根节点的子树结果
	wordList := getAllChildrenNodeData(node)
	words := sortAndFilter(wordList)
	if len(words) > 10 {
		return words[:10]
	}
	return words
}

// getAllChildrenNodeData 遍历以 node 为根节点的所有 Trie 子树
func getAllChildrenNodeData(node *Trie) (wordList wordInfos) {
	// 已经找到一个有效拼音
	if node.IsWord {
		wordList = append(wordList, node.Words...)
	}

	for _, child := range node.Children {
		if child == nil {
			continue
		}
		// 取有效子树的结果
		wordList = append(wordList, getAllChildrenNodeData(child)...)
	}
	return wordList
}

// sortAndFilter 获取汉字 List（排序 & 去重）
func sortAndFilter(wordList wordInfos) []string {
	// 对wordList进行排序 (优先分数，分数相同按照拼音序)
	sort.Stable(wordList)
	// 去重 & 读取结果
	hashMap := make(map[string]bool, len(wordList))
	words := make([]string, 0, len(wordList))
	for _, w := range wordList {
		_, ok := hashMap[w.word]
		if ok {
			continue
		}
		hashMap[w.word] = true
		words = append(words, w.word)
	}
	return words
}
