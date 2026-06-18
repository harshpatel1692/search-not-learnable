window.TRACES = [
 {
  "id": "3e5c7d9b",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "937",
  "line": "s16: x != a*b+1: EQ1 ones: a ends 2, b ends 1, result ends E in {3,4,5,6,7,8,9}; 2*1 then +1 ends 3 -> none matches -> drop.",
  "why": "1 stated pair(s) actually match the target"
 },
 {
  "id": "9346686a",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "844",
  "line": "s13: y != a*b+1: EQ2 ones: a ends D in {2,3}, b ends 1, result ends G in {4,5}; 2*1 then +1 ends 3; 3*1 then +1 ends 4 -> none matches -> drop.",
  "why": "1 stated pair(s) actually match the target"
 },
 {
  "id": "2c9a8df6",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "656",
  "line": "s11: z != a*b: EQ3 ones: a ends E in {2,3,4}, b ends 1, result ends a's ones symbol; 2*1 ends 2; 3*1 ends 3; 4*1 ends 4 -> none matches -> drop.",
  "why": "3 stated pair(s) actually match the target"
 },
 {
  "id": "610bf536",
  "type": "kill",
  "family": "onesenum",
  "axis": "transcription",
  "tok": "770",
  "line": "s13: x != a+b: EQ1 ones: a ends B in {3,4,5,6}, b ends 3, result ends E in {4,5,6,7,8,9}; 3+4 ends 7; 3+5 ends 8; 3+6 ends 9 -> none matches -> drop.",
  "why": "b-ones digit 3 != pin of C (None) | 3 stated pair(s) actually match the target | pair enumeration incomplete vs stated domains"
 },
 {
  "id": "2ff200fb",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "727",
  "line": "s11: x != a*b-1: EQ1 ones: a ends B in {2,3}, b ends D in {2,3}, result ends G in {4,5,6}; 2*3 then -1 ends 5; 3*2 then -1 ends 5 -> none matches -> drop.",
  "why": "2 stated pair(s) actually match the target"
 },
 {
  "id": "42bde66c",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "1112",
  "line": "s19: x != a*b+1: EQ1 ones: a ends B in {3,4}, b ends D in {3,4}, result ends a's ones symbol; 3*4 then +1 ends 3; 4*3 then +1 ends 3 -> none matches -> drop.",
  "why": "1 stated pair(s) actually match the target"
 },
 {
  "id": "6beb3a1f",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "675",
  "line": "s11: x != a*b: EQ1 ones: a ends B in {2,3,4}, b ends C in {2,3}, result ends F in {4,5,6}; 2*3 ends 6; 3*2 ends 6; 4*2 ends 8; 4*3 ends 2 -> none matches -> drop.",
  "why": "2 stated pair(s) actually match the target"
 },
 {
  "id": "a12f00f4",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "774",
  "line": "s14: z != a*b: EQ3 ones: a ends C in {3,4}, b ends C in {3,4}, result ends H in {4,5,6,7,8,9}; 3*3 ends 9; 3*4 ends 2; 4*3 ends 2; 4*4 ends 6 -> none matches -> drop.",
  "why": "2 stated pair(s) actually match the target"
 },
 {
  "id": "3b206148",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "655",
  "line": "s6: x != a+b-1: EQ5 ones: a ends B in {1,2}, b ends H in {2,3}, result ends I in {1,2,3,4,5,6,7,8,9}; 1+2 then -1 ends 2; 1+3 then -1 ends 3; 2+3 then -1 ends 4 -> none matches -> drop.",
  "why": "3 stated pair(s) actually match the target"
 },
 {
  "id": "35a562bd",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "729",
  "line": "s13: x != a+b: EQ1 ones: a ends A in {3,4,5,6,7,8,9}, b ends 2, result ends D=1; 3+2 ends 5; 4+2 ends 6; 5+2 ends 7; 6+2 ends 8; 7+2 ends 9; 8+2 ends 0; 9+2 ends 1 -> none matches -> drop.",
  "why": "1 stated pair(s) actually match the target"
 },
 {
  "id": "ae4aef23",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "846",
  "line": "s15: x != a+b: EQ3 ones: a ends 1, b ends J in {4,5,6,7,8,9}, result ends D in {4,5,6,7,8,9}; 1+4 ends 5; 1+5 ends 6; 1+6 ends 7; 1+7 ends 8; 1+8 ends 9; 1+9 ends 0 -> none matches -> drop.",
  "why": "5 stated pair(s) actually match the target"
 },
 {
  "id": "df3262bc",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "685",
  "line": "s11: y != a*b: EQ3 ones: a ends D in {2,3,4}, b ends C in {2,3,4}, result ends A in {6,7,8,9}; 2*3 ends 6; 2*4 ends 8; 3*2 ends 6; 3*4 ends 2; 4*2 ends 8; 4*3 ends 2 -> none matches -> drop.",
  "why": "4 stated pair(s) actually match the target"
 },
 {
  "id": "a266aeb5",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "724",
  "line": "s13: y != a*b: EQ3 ones: a ends A in {3,4,5,6,7,8,9}, b ends 1, result ends a's ones symbol; 3*1 ends 3; 4*1 ends 4; 5*1 ends 5; 6*1 ends 6; 7*1 ends 7; 8*1 ends 8; 9*1 ends 9 -> none matches -> drop.",
  "why": "7 stated pair(s) actually match the target"
 },
 {
  "id": "9eaae1f1",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "651",
  "line": "s11: x != a*b: EQ1 ones: a ends 1, b ends B in {2,3,4,5,6,7,8,9}, result ends b's ones symbol; 1*2 ends 2; 1*3 ends 3; 1*4 ends 4; 1*5 ends 5; 1*6 ends 6; 1*7 ends 7; 1*8 ends 8; 1*9 ends 9 -> none matches -> drop.",
  "why": "8 stated pair(s) actually match the target"
 },
 {
  "id": "2d89386e",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "722",
  "line": "s11: z != a*b: EQ3 ones: a ends 1, b ends E in {2,3,4,5,6,7,8,9}, result ends A in {2,3,4,5,6,7,8,9}; 1*2 ends 2; 1*3 ends 3; 1*4 ends 4; 1*5 ends 5; 1*6 ends 6; 1*7 ends 7; 1*8 ends 8; 1*9 ends 9 -> none matches -> drop.",
  "why": "8 stated pair(s) actually match the target"
 },
 {
  "id": "46fcfa9c",
  "type": "kill",
  "family": "onesenum",
  "axis": "verdict",
  "tok": "758",
  "line": "s11: x != a*b: EQ1 ones: a ends 1, b ends C in {2,3,4,5,6,7,8,9}, result ends D in {2,3,4,5,6,7,8,9}; 1*2 ends 2; 1*3 ends 3; 1*4 ends 4; 1*5 ends 5; 1*6 ends 6; 1*7 ends 7; 1*8 ends 8; 1*9 ends 9 -> none matches -> drop.",
  "why": "8 stated pair(s) actually match the target"
 }
];
