
class TyposKB:
    """
    Keyboard Algorithm

    Methods
    --------
    __keyboard_dic__
    __search_kb_key__
    __change_char__
    __getTypos_based_kb__
    """
    def __init__(self):
        self.array_prox= self.__keyboard_dic__()

    def __keyboard_dic__(self):
        """
        Create the dictionary for based on the keyboard (German keyboard)
        without the characters: ü,ö,ä and ß
        """
        array_prox = {}
        array_prox['a'] = ['q', 'w', 'y', 'x']
        array_prox['b'] = ['v', 'f', 'g', 'h', 'n']
        array_prox['c'] = ['x', 's', 'd', 'f', 'v']
        array_prox['d'] = ['x', 's', 'w', 'e', 'r', 'f', 'v', 'c']
        array_prox['e'] = ['w', 's', 'd', 'f', 'r']
        array_prox['f'] = ['c', 'd', 'e', 'r', 't', 'g', 'b', 'v']
        array_prox['g'] = ['r', 'f', 'v', 't', 'b', 'z', 'h', 'n']
        array_prox['h'] = ['b', 'g', 't', 'z', 'u', 'j', 'm', 'n']
        array_prox['i'] = ['u', 'j', 'k', 'l', 'o']
        array_prox['j'] = ['n', 'h', 'z', 'u', 'i', 'k', 'm']
        array_prox['k'] = ['u', 'j', 'm', 'l', 'o']
        array_prox['l'] = ['p', 'o', 'i', 'k', 'm']
        array_prox['m'] = ['n', 'h', 'j', 'k', 'l']
        array_prox['n'] = ['b', 'g', 'h', 'j', 'm']
        array_prox['o'] = ['i', 'k', 'l', 'p']
        array_prox['p'] = ['o', 'l']
        array_prox['r'] = ['e', 'd', 'f', 'g', 't']
        array_prox['s'] = ['q', 'w', 'e', 'y', 'x', 'c']
        array_prox['t'] = ['r', 'f', 'g', 'h', 'z']
        array_prox['u'] = ['z', 'h', 'j', 'k', 'i']
        array_prox['v'] = ['', 'c', 'd', 'f', 'g', 'b']    
        array_prox['w'] = ['q', 'a', 's', 'd', 'e']
        array_prox['x'] = ['y', 'a', 's', 'd', 'c']
        array_prox['z'] = ['t', 'g', 'h', 'j', 'u']
        array_prox['y'] = ['x', 's', 'a']
        array_prox['1'] = ['q', 'w']
        array_prox['2'] = ['q', 'w', 'e']
        array_prox['3'] = ['w', 'e', 'r']
        array_prox['4'] = ['e', 'r', 't']
        array_prox['5'] = ['r', 't', 'z']
        array_prox['6'] = ['t', 'z', 'u']
        array_prox['7'] = ['z', 'u', 'i']
        array_prox['8'] = ['u', 'i', 'o']
        array_prox['9'] = ['i', 'o', 'p']
        array_prox['0'] = ['o', 'p']
        return array_prox

    def __search_kb_key__(self, key):
        """
        Search for key in the keyboard dictionary
        Parameters
        ----------
        key:str
            the character to be changed
        Return
        ------
        str:
            the key if it is not found in the dictionary or the array based on the key
        """
        if key not in self.array_prox:
            return key
        return self.array_prox[key]

    def change_char(self,s, p, r):
        return s[:p]+r+s[p+1:]

    def getTypos_based_kb(self,word_to_change):
        """
        Parameters
        ----------
        word_to_change:str
            the word to be modified 
        Return
        ------

        """
        arr = []
        for i,char_w in enumerate(word_to_change):
            temp = self.__search_kb_key__(char_w)
            if len(temp) <= 1:
                continue
            for char_t in temp:
                typo =self.change_char(word_to_change,i,char_t)
                arr.append(typo)
        return arr

