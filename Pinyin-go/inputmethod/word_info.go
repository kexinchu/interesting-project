package inputmethod

// wordInfo 数据结构
type wordInfo struct {
	spell string
	word  string
	frac  int // 分数
}

// 针对 []wordInfo 实现稳定排序
type wordInfos []wordInfo

// Len 重写 Len() 方法
func (a wordInfos) Len() int {
	return len(a)
}

// Swap 重写 Swap() 方法
func (a wordInfos) Swap(i int, j int) {
	a[i], a[j] = a[j], a[i]
}

// Less 重写 Less() 方法
func (a wordInfos) Less(i int, j int) bool {
	// 首先高分在前
	if a[i].frac > a[j].frac {
		return true
	}
	// 同分 => 字母序小的在前
	if a[i].frac == a[j].frac {
		return a[i].spell < a[j].spell
	}
	return false
}
