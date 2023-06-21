package inputmethod

import (
	"testing"
)

func TestWordInfos_Len(tt *testing.T) {
	tests := []struct {
		name  string
		words wordInfos
		want  int
	}{
		{
			name: "case1",
			words: wordInfos{
				{spell: "dong", word: "东", frac: 9},
				{spell: "dong", word: "懂", frac: 8},
			},
			want: 2,
		},
	}
	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			if got := test.words.Len(); got != test.want {
				tt.Errorf("Len() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestWordInfos_Less(tt *testing.T) {
	tests := []struct {
		name  string
		words wordInfos
		index [2]int
		want  bool
	}{
		{
			name:  "case1",
			words: []wordInfo{{spell: "dong", word: "东", frac: 9}, {spell: "dong", word: "懂", frac: 8}},
			index: [2]int{0, 1},
			want:  true,
		},
		{
			name:  "case2",
			words: []wordInfo{{spell: "dong", word: "懂", frac: 8}, {spell: "dong", word: "东", frac: 9}},
			index: [2]int{0, 1},
			want:  false,
		},
		{
			name:  "case3",
			words: []wordInfo{{spell: "na", word: "那", frac: 9}, {spell: "nan", word: "南", frac: 9}},
			index: [2]int{0, 1},
			want:  true,
		},
		{
			name:  "case4",
			words: []wordInfo{{spell: "na", word: "那", frac: 9}, {spell: "nan", word: "难", frac: 8}},
			index: [2]int{0, 1},
			want:  true,
		},
		{
			name:  "case5",
			words: []wordInfo{{spell: "na", word: "娜", frac: 7}, {spell: "nan", word: "难", frac: 8}},
			index: [2]int{0, 1},
			want:  false,
		},
	}
	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			if got := test.words.Less(test.index[0], test.index[1]); got != test.want {
				tt.Errorf("Less() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestWordInfos_Swap(tt *testing.T) {
	tests := []struct {
		name  string
		words wordInfos
		index [2]int
	}{
		{
			name:  "case1",
			words: wordInfos{{spell: "dong", word: "东", frac: 9}, {spell: "dong", word: "懂", frac: 8}},
			index: [2]int{0, 1},
		},
	}
	for _, test := range tests {
		tt.Run(test.name, func(tt *testing.T) {
			test.words.Swap(test.index[0], test.index[1])
		})
	}
}
