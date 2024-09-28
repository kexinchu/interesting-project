package inputmethod

import (
	"reflect"
	"testing"
)

func TestSpellTrie_insert(tt *testing.T) {
	t := &Trie{}
	tests := []struct {
		name  string
		spell string
		words wordInfos
	}{
		{
			name:  "case1",
			spell: "dong",
			words: wordInfos{
				{spell: "dong", word: "东", frac: 9},
				{spell: "dong", word: "懂", frac: 8},
				{spell: "dong", word: "动", frac: 7},
				{spell: "dong", word: "洞", frac: 9},
			},
		},
		{
			name:  "case2",
			spell: "-dong",
			words: wordInfos{
				{spell: "dong", word: "东", frac: 9},
				{spell: "dong", word: "懂", frac: 8},
				{spell: "dong", word: "动", frac: 7},
				{spell: "dong", word: "洞", frac: 9},
			},
		},
	}

	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			t.insert(test.spell, test.words)
		})
	}
}

func getSampleTrie() *Trie {
	// 构造Trie树
	t := &Trie{}
	samples := []struct {
		spell string
		words wordInfos
	}{
		{
			spell: "na",
			words: wordInfos{
				{spell: "na", word: "那", frac: 9},
				{spell: "na", word: "拿", frac: 9},
				{spell: "na", word: "哪", frac: 8},
				{spell: "na", word: "呐", frac: 7},
				{spell: "na", word: "娜", frac: 7},
			},
		},
		{
			spell: "nan",
			words: wordInfos{
				{spell: "nan", word: "南", frac: 9},
				{spell: "nan", word: "男", frac: 9},
				{spell: "nan", word: "难", frac: 8},
				{spell: "nan", word: "楠", frac: 7},
			},
		},
	}
	for _, sample := range samples {
		t.insert(sample.spell, sample.words)
	}
	return t
}

func TestSpellTrie_search(tt *testing.T) {
	// 构造Trie树
	t := getSampleTrie()

	tests := []struct {
		name  string
		spell string
		want  wordInfos
	}{
		{
			name:  "case1",
			spell: "nan",
			want: wordInfos{
				{spell: "nan", word: "南", frac: 9},
				{spell: "nan", word: "男", frac: 9},
				{spell: "nan", word: "难", frac: 8},
				{spell: "nan", word: "楠", frac: 7},
			},
		},
		{
			name:  "case2",
			spell: "-",
			want:  nil,
		},
		{
			name:  "case3",
			spell: "n",
			want:  nil,
		},
		{
			name:  "case4",
			spell: "na",
			want: wordInfos{
				{spell: "na", word: "那", frac: 9},
				{spell: "na", word: "拿", frac: 9},
				{spell: "na", word: "哪", frac: 8},
				{spell: "na", word: "呐", frac: 7},
				{spell: "na", word: "娜", frac: 7},
			},
		},
	}

	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			get, ok := t.search(test.spell)
			if ok && !reflect.DeepEqual(get.Words, test.want) {
				tt.Errorf("search() = %v, want %v", get.Words, test.want)
			}
		})
	}
}

func TestSpellTrie_getWords(tt *testing.T) {
	// 构造Trie树
	t := getSampleTrie()

	tests := []struct {
		name  string
		spell string
		want  []string
	}{
		{
			name:  "case1",
			spell: "nan",
			want:  []string{"南", "男", "难", "楠"},
		},
		{
			name:  "case2",
			spell: "-",
			want:  nil,
		},
		{
			name:  "case3",
			spell: "n",
			want:  []string{"那", "拿", "南", "男", "哪", "难", "呐", "娜", "楠"},
		},
		{
			name:  "case4",
			spell: "na",
			want:  []string{"那", "拿", "哪", "呐", "娜"},
		},
	}

	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			node, ok := t.search(test.spell)
			if ok {
				if get := getWords(node); !reflect.DeepEqual(get, test.want) {
					tt.Errorf("getWords() = %v, want %v", get, test.want)
				}
			}
		})
	}
}

func TestSpellTrie_getAllChildrenNodeData(tt *testing.T) {
	// 构造Trie树
	t := getSampleTrie()

	tests := []struct {
		name  string
		spell string
		want  wordInfos
	}{
		{
			name:  "case1",
			spell: "n",
			want: wordInfos{
				{spell: "na", word: "那", frac: 9},
				{spell: "na", word: "拿", frac: 9},
				{spell: "na", word: "哪", frac: 8},
				{spell: "na", word: "呐", frac: 7},
				{spell: "na", word: "娜", frac: 7},
				{spell: "nan", word: "南", frac: 9},
				{spell: "nan", word: "男", frac: 9},
				{spell: "nan", word: "难", frac: 8},
				{spell: "nan", word: "楠", frac: 7},
			},
		},
	}
	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			node, ok := t.search(test.spell)
			words := getAllChildrenNodeData(node)
			if ok && !reflect.DeepEqual(words, test.want) {
				tt.Errorf("search() = %v, want %v", words, test.want)
			}
		})
	}
}

func TestSpellTrie_sortAndFilter(tt *testing.T) {
	tests := []struct {
		name  string
		words wordInfos
		want  []string
	}{
		{
			name: "case1",
			words: wordInfos{
				{spell: "na", word: "那", frac: 9},
				{spell: "na", word: "拿", frac: 9},
				{spell: "na", word: "哪", frac: 8},
				{spell: "na", word: "呐", frac: 7},
				{spell: "na", word: "娜", frac: 7},
				{spell: "nan", word: "南", frac: 9},
				{spell: "nan", word: "男", frac: 9},
				{spell: "nan", word: "难", frac: 8},
				{spell: "nan", word: "楠", frac: 7},
			},
			want: []string{"那", "拿", "南", "男", "哪", "难", "呐", "娜", "楠"},
		},
	}
	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			words := sortAndFilter(test.words)
			if !reflect.DeepEqual(words, test.want) {
				tt.Errorf("search() = %v, want %v", words, test.want)
			}
		})
	}
}
