class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        
        self.root = TrieNode()
    
    def insert(self,word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            
            node = node.children[char]
        

        node.is_end = True
    
    def search(self,word:str):
        node = self.root
        if word.strip() == "":
            return False
        for char in word:
            if char not in node.children:
                return False
            
            node = node.children[char]
        return True
    

trie = Trie()
# trie.insert("Hello")
# words = ["search", "sea", "shore", "complete", "company"]
# print(trie.search(""))
# print(trie.search(words[0]))
# for i in words:
#     trie.insert(i)


# print(trie.search('searchhhh'))
# print(trie.search('sea'))
