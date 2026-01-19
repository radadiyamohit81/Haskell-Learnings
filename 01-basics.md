# Haskell Learning Path - Part 1: Basics

## 1. What is Haskell?

Haskell is a **purely functional**, **statically typed**, **lazy** programming language.

- **Purely Functional**: Functions have no side effects, same input always gives same output
- **Statically Typed**: Types are checked at compile time
- **Lazy Evaluation**: Values are computed only when needed

---

## 2. Basic Syntax

### 2.1 Variables (Immutable Bindings)

```haskell
-- Variables are immutable (can't change once set)
name :: String
name = "Mohit"

age :: Int
age = 25

pi' :: Double
pi' = 3.14159
```

### 2.2 Basic Types

```haskell
-- Integers
x :: Int          -- Fixed precision integer
y :: Integer      -- Arbitrary precision integer

-- Floating point
a :: Float        -- Single precision
b :: Double       -- Double precision

-- Boolean
flag :: Bool
flag = True       -- or False

-- Character and String
char :: Char
char = 'A'

str :: String     -- String is just [Char]
str = "Hello"

-- Lists (homogeneous - all same type)
nums :: [Int]
nums = [1, 2, 3, 4, 5]

-- Tuples (heterogeneous - different types allowed)
person :: (String, Int)
person = ("Mohit", 25)
```

### 2.3 Type Inference

```haskell
-- Haskell can infer types (but explicit is better for readability)
greeting = "Hello"  -- Haskell infers :: String
number = 42         -- Haskell infers :: Num a => a (polymorphic)
```

---

## 3. Functions

### 3.1 Defining Functions

```haskell
-- Function with type signature
add :: Int -> Int -> Int
add x y = x + y

-- Usage
result = add 3 5  -- result = 8

-- Note: No parentheses for arguments!
-- add(3, 5)  ❌ Wrong
-- add 3 5    ✅ Correct
```

### 3.2 Function Application

```haskell
-- Function application is left-associative
-- f a b c = ((f a) b) c

double :: Int -> Int
double x = x * 2

triple :: Int -> Int
triple x = x * 3

-- Composition with $
result1 = double (triple 5)   -- 30
result2 = double $ triple 5   -- 30 (same, $ avoids parentheses)

-- Function composition with .
doubleThenTriple :: Int -> Int
doubleThenTriple = triple . double  -- Read right to left: double first, then triple

result3 = doubleThenTriple 5  -- 30
```

### 3.3 Lambda Functions (Anonymous Functions)

```haskell
-- Lambda syntax: \args -> expression
addLambda :: Int -> Int -> Int
addLambda = \x y -> x + y

-- Common in higher-order functions
doubled = map (\x -> x * 2) [1, 2, 3]  -- [2, 4, 6]
```

### 3.4 Pattern Matching

```haskell
-- Pattern matching in function definitions
factorial :: Int -> Int
factorial 0 = 1                    -- Base case
factorial n = n * factorial (n - 1)  -- Recursive case

-- Pattern matching on lists
head' :: [a] -> a
head' []    = error "Empty list"
head' (x:_) = x   -- x:xs means x is head, xs is tail

tail' :: [a] -> [a]
tail' []     = error "Empty list"
tail' (_:xs) = xs

-- Pattern matching on tuples
fst' :: (a, b) -> a
fst' (x, _) = x

snd' :: (a, b) -> b
snd' (_, y) = y
```

### 3.5 Guards

```haskell
-- Guards are like if-else chains
absoluteValue :: Int -> Int
absoluteValue n
  | n < 0     = -n
  | otherwise = n

grade :: Int -> String
grade score
  | score >= 90 = "A"
  | score >= 80 = "B"
  | score >= 70 = "C"
  | score >= 60 = "D"
  | otherwise   = "F"
```

### 3.6 Where and Let

```haskell
-- 'where' binds at end of function
quadraticRoots :: Double -> Double -> Double -> (Double, Double)
quadraticRoots a b c = (root1, root2)
  where
    discriminant = b * b - 4 * a * c
    root1 = (-b + sqrt discriminant) / (2 * a)
    root2 = (-b - sqrt discriminant) / (2 * a)

-- 'let' binds at beginning (expression-based)
quadraticRoots' :: Double -> Double -> Double -> (Double, Double)
quadraticRoots' a b c =
  let discriminant = b * b - 4 * a * c
      root1 = (-b + sqrt discriminant) / (2 * a)
      root2 = (-b - sqrt discriminant) / (2 * a)
  in (root1, root2)
```

---

## 4. Lists

### 4.1 List Operations

```haskell
-- List construction
list1 = [1, 2, 3, 4, 5]
list2 = 1 : 2 : 3 : []   -- Same as [1, 2, 3], : is "cons"
list3 = [1..10]          -- [1,2,3,4,5,6,7,8,9,10]
list4 = [1,3..10]        -- [1,3,5,7,9] (step of 2)

-- Basic operations
length [1, 2, 3]         -- 3
head [1, 2, 3]           -- 1
tail [1, 2, 3]           -- [2, 3]
init [1, 2, 3]           -- [1, 2]
last [1, 2, 3]           -- 3
null []                  -- True
null [1]                 -- False

-- Concatenation
[1, 2] ++ [3, 4]         -- [1, 2, 3, 4]

-- Indexing (0-based)
[1, 2, 3] !! 1           -- 2

-- Take and Drop
take 3 [1, 2, 3, 4, 5]   -- [1, 2, 3]
drop 3 [1, 2, 3, 4, 5]   -- [4, 5]

-- Reverse
reverse [1, 2, 3]        -- [3, 2, 1]
```

### 4.2 List Comprehensions

```haskell
-- Basic comprehension
squares = [x * x | x <- [1..10]]
-- [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

-- With predicate (filter)
evenSquares = [x * x | x <- [1..10], even x]
-- [4, 16, 36, 64, 100]

-- Multiple generators
pairs = [(x, y) | x <- [1, 2, 3], y <- ['a', 'b']]
-- [(1,'a'),(1,'b'),(2,'a'),(2,'b'),(3,'a'),(3,'b')]

-- Nested with conditions
pythagorean = [(a, b, c) | c <- [1..20], b <- [1..c], a <- [1..b], a*a + b*b == c*c]
-- [(3,4,5),(6,8,10),(5,12,13),(9,12,15),(8,15,17),(12,16,20)]
```

---

## 5. Higher-Order Functions

Functions that take functions as arguments or return functions.

### 5.1 Map

```haskell
-- Apply function to each element
map :: (a -> b) -> [a] -> [b]

map (*2) [1, 2, 3]           -- [2, 4, 6]
map show [1, 2, 3]           -- ["1", "2", "3"]
map length ["hello", "hi"]   -- [5, 2]
```

### 5.2 Filter

```haskell
-- Keep elements that satisfy predicate
filter :: (a -> Bool) -> [a] -> [a]

filter even [1, 2, 3, 4, 5]        -- [2, 4]
filter (> 3) [1, 2, 3, 4, 5]       -- [4, 5]
filter (/= ' ') "hello world"     -- "helloworld"
```

### 5.3 Fold (Reduce)

```haskell
-- Reduce list to single value
-- foldl: left fold (start from left)
-- foldr: right fold (start from right)

foldl :: (b -> a -> b) -> b -> [a] -> b
foldr :: (a -> b -> b) -> b -> [a] -> b

-- Sum with foldl
sum' = foldl (+) 0 [1, 2, 3, 4]  -- ((((0+1)+2)+3)+4) = 10

-- Product with foldr
product' = foldr (*) 1 [1, 2, 3, 4]  -- 1*(2*(3*(4*1))) = 24

-- Implementing common functions with fold
length' = foldl (\acc _ -> acc + 1) 0
reverse' = foldl (\acc x -> x : acc) []
map' f = foldr (\x acc -> f x : acc) []
filter' p = foldr (\x acc -> if p x then x : acc else acc) []
```

### 5.4 Currying

All functions in Haskell are curried (take one argument at a time).

```haskell
-- add :: Int -> Int -> Int
-- is actually
-- add :: Int -> (Int -> Int)

add :: Int -> Int -> Int
add x y = x + y

-- Partial application
add5 :: Int -> Int
add5 = add 5    -- Partially applied, waiting for second argument

result = add5 3  -- 8

-- More examples
multiplyBy2 = (*) 2
greaterThan10 = (> 10)
appendHello = ("Hello " ++)

multiplyBy2 5      -- 10
greaterThan10 15   -- True
appendHello "World" -- "Hello World"
```

---

## 6. Practice Exercises

```haskell
-- Exercise 1: Implement your own 'sum' using recursion
mySum :: [Int] -> Int
mySum []     = 0
mySum (x:xs) = x + mySum xs

-- Exercise 2: Implement 'elem' (check if element exists in list)
myElem :: Eq a => a -> [a] -> Bool
myElem _ []     = False
myElem e (x:xs) = e == x || myElem e xs

-- Exercise 3: Implement 'zip'
myZip :: [a] -> [b] -> [(a, b)]
myZip [] _          = []
myZip _ []          = []
myZip (x:xs) (y:ys) = (x, y) : myZip xs ys

-- Exercise 4: Implement 'takeWhile'
myTakeWhile :: (a -> Bool) -> [a] -> [a]
myTakeWhile _ []     = []
myTakeWhile p (x:xs)
  | p x       = x : myTakeWhile p xs
  | otherwise = []
```

---

## Next: [Part 2 - Type System & Custom Types](./02-types.md)
