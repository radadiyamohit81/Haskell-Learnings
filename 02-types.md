# Haskell Learning Path - Part 2: Type System & Custom Types

## 1. Type Signatures Explained

### 1.1 Reading Type Signatures

```haskell
-- Format: functionName :: InputType -> OutputType

-- Single argument
not :: Bool -> Bool

-- Multiple arguments (curried)
add :: Int -> Int -> Int
-- Read as: Int -> (Int -> Int)
-- Takes Int, returns function that takes Int and returns Int

-- Polymorphic (generic) types use lowercase
id :: a -> a              -- Any type 'a', returns same type
const :: a -> b -> a      -- Takes any 'a', any 'b', returns 'a'
```

### 1.2 Type Variables

```haskell
-- 'a', 'b', 'c' are type variables (like generics in other languages)
fst :: (a, b) -> a        -- Works for any types a and b
snd :: (a, b) -> b

head :: [a] -> a          -- Works for list of any type
tail :: [a] -> [a]

map :: (a -> b) -> [a] -> [b]
-- Takes: function from a to b, list of a
-- Returns: list of b
```

---

## 2. Type Classes (Interfaces/Traits)

Type classes define behavior that types can implement.

### 2.1 Common Type Classes

```haskell
-- Eq: Types that can be compared for equality
class Eq a where
  (==) :: a -> a -> Bool
  (/=) :: a -> a -> Bool

-- Ord: Types that can be ordered (extends Eq)
class Eq a => Ord a where
  compare :: a -> a -> Ordering  -- LT | EQ | GT
  (<), (<=), (>), (>=) :: a -> a -> Bool
  max, min :: a -> a -> a

-- Show: Types that can be converted to String
class Show a where
  show :: a -> String

-- Read: Types that can be parsed from String
class Read a where
  read :: String -> a

-- Num: Numeric types
class Num a where
  (+), (-), (*) :: a -> a -> a
  negate, abs, signum :: a -> a
  fromInteger :: Integer -> a

-- Functor: Types that can be mapped over
class Functor f where
  fmap :: (a -> b) -> f a -> f b
```

### 2.2 Type Class Constraints

```haskell
-- Functions can require types to implement certain classes
-- Constraint => Type

-- Requires 'a' to be Eq (can use ==)
elem :: Eq a => a -> [a] -> Bool
elem _ []     = False
elem e (x:xs) = e == x || elem e xs

-- Requires 'a' to be Ord (can use <, >, compare)
sort :: Ord a => [a] -> [a]

-- Multiple constraints
showAndCompare :: (Show a, Ord a) => a -> a -> String
showAndCompare x y = show x ++ " vs " ++ show y ++ ": " ++ 
  case compare x y of
    LT -> "first is less"
    EQ -> "equal"
    GT -> "first is greater"
```

---

## 3. Custom Data Types

### 3.1 Type Aliases (type)

```haskell
-- Just a nickname, interchangeable with original
type String = [Char]  -- This is how String is defined in Prelude

type Name = String
type Age = Int
type Person = (Name, Age)

greet :: Name -> String
greet name = "Hello, " ++ name

-- Can use String where Name is expected and vice versa
greeting = greet "Mohit"  -- Works fine
```

### 3.2 New Types (newtype)

```haskell
-- Wrapper with single constructor and single field
-- Zero runtime overhead (erased at compile time)
-- Creates a distinct type (not interchangeable)

newtype CustomerId = CustomerId Int
  deriving (Show, Eq)

newtype Email = Email String
  deriving (Show, Eq)

-- Can't accidentally mix up CustomerId with regular Int
processCustomer :: CustomerId -> String
processCustomer (CustomerId id) = "Processing customer: " ++ show id

-- This won't compile:
-- processCustomer 42  -- Error! Expected CustomerId, got Int

-- Must wrap explicitly:
processCustomer (CustomerId 42)  -- Works!
```

### 3.3 Algebraic Data Types (data)

#### Sum Types (OR - multiple constructors)

```haskell
-- Bool is a sum type with two constructors
data Bool = False | True

-- Custom status type
data OrderStatus 
  = Pending
  | Processing
  | Shipped
  | Delivered
  | Cancelled
  deriving (Show, Eq)

-- Pattern match on constructors
statusMessage :: OrderStatus -> String
statusMessage Pending    = "Order is pending"
statusMessage Processing = "Order is being processed"
statusMessage Shipped    = "Order has been shipped"
statusMessage Delivered  = "Order delivered!"
statusMessage Cancelled  = "Order was cancelled"
```

#### Product Types (AND - single constructor with multiple fields)

```haskell
-- Product type: contains multiple fields
data Point = Point Double Double
  deriving (Show, Eq)

-- Record syntax (named fields)
data Person = Person
  { personName :: String
  , personAge  :: Int
  , personEmail :: String
  }
  deriving (Show, Eq)

-- Creating instances
point1 = Point 3.0 4.0
person1 = Person "Mohit" 25 "mohit@example.com"
person2 = Person { personName = "John", personAge = 30, personEmail = "john@example.com" }

-- Accessing fields
getName :: Person -> String
getName person = personName person

-- Record update syntax (creates new record)
olderPerson = person1 { personAge = 26 }
```

#### Combined Sum + Product Types

```haskell
-- Real-world example: Payment method
data PaymentMethod
  = Cash
  | Card { cardNumber :: String, cardExpiry :: String }
  | UPI { upiId :: String }
  | Wallet { walletProvider :: String, walletBalance :: Double }
  deriving (Show, Eq)

processPayment :: PaymentMethod -> Double -> String
processPayment Cash amount = 
  "Received " ++ show amount ++ " in cash"
processPayment (Card num _) amount = 
  "Charged " ++ show amount ++ " to card ending in " ++ take 4 (reverse num)
processPayment (UPI upi) amount = 
  "Debited " ++ show amount ++ " from UPI: " ++ upi
processPayment (Wallet provider balance) amount
  | balance >= amount = "Paid " ++ show amount ++ " via " ++ provider
  | otherwise = "Insufficient " ++ provider ++ " balance"
```

### 3.4 Parameterized Types (Generics)

```haskell
-- Maybe: represents optional values
data Maybe a = Nothing | Just a
  deriving (Show, Eq)

-- Either: represents success or failure
data Either a b = Left a | Right b
  deriving (Show, Eq)

-- Custom examples
data Box a = Box a
  deriving (Show)

data Pair a b = Pair a b
  deriving (Show)

data Tree a
  = Leaf a
  | Node (Tree a) a (Tree a)
  deriving (Show)

-- Usage
box1 :: Box Int
box1 = Box 42

box2 :: Box String
box2 = Box "hello"

pair1 :: Pair String Int
pair1 = Pair "age" 25

tree1 :: Tree Int
tree1 = Node (Leaf 1) 2 (Leaf 3)
--        1
--       /|\
--      2
--       \
--        3
```

---

## 4. Working with Maybe

```haskell
-- Maybe represents optional/nullable values
-- data Maybe a = Nothing | Just a

-- Safe head function
safeHead :: [a] -> Maybe a
safeHead []    = Nothing
safeHead (x:_) = Just x

-- Safe division
safeDiv :: Int -> Int -> Maybe Int
safeDiv _ 0 = Nothing
safeDiv x y = Just (x `div` y)

-- Lookup in association list
lookup' :: Eq k => k -> [(k, v)] -> Maybe v
lookup' _ []          = Nothing
lookup' key ((k,v):rest)
  | key == k  = Just v
  | otherwise = lookup' key rest

-- Handling Maybe values
displayResult :: Maybe Int -> String
displayResult Nothing  = "No result"
displayResult (Just x) = "Result: " ++ show x

-- Using 'maybe' function
-- maybe :: b -> (a -> b) -> Maybe a -> b
result = maybe "default" show (Just 42)  -- "42"
result2 = maybe "default" show Nothing   -- "default"

-- Using 'fromMaybe'
-- fromMaybe :: a -> Maybe a -> a
import Data.Maybe (fromMaybe)
value = fromMaybe 0 (Just 42)  -- 42
value2 = fromMaybe 0 Nothing   -- 0
```

---

## 5. Working with Either

```haskell
-- Either represents success (Right) or failure (Left)
-- data Either a b = Left a | Right b
-- Convention: Left = error, Right = success

data ValidationError
  = EmptyName
  | InvalidAge
  | InvalidEmail
  deriving (Show)

validateName :: String -> Either ValidationError String
validateName "" = Left EmptyName
validateName name = Right name

validateAge :: Int -> Either ValidationError Int
validateAge age
  | age < 0 || age > 150 = Left InvalidAge
  | otherwise = Right age

validateEmail :: String -> Either ValidationError String
validateEmail email
  | '@' `elem` email = Right email
  | otherwise = Left InvalidEmail

-- Using either function
-- either :: (a -> c) -> (b -> c) -> Either a b -> c
handleResult :: Either ValidationError String -> String
handleResult = either 
  (\err -> "Error: " ++ show err)
  (\val -> "Success: " ++ val)
```

---

## 6. Deriving Type Classes

```haskell
-- Haskell can automatically derive common type classes

data Color = Red | Green | Blue
  deriving (
    Show,     -- Can convert to String
    Eq,       -- Can compare with ==
    Ord,      -- Can compare with <, >, etc (order based on definition)
    Enum,     -- Can enumerate: [Red .. Blue]
    Bounded   -- Has minBound and maxBound
  )

-- Usage
show Red           -- "Red"
Red == Red         -- True
Red < Green        -- True (based on definition order)
[Red .. Blue]      -- [Red, Green, Blue]
minBound :: Color  -- Red
maxBound :: Color  -- Blue
```

---

## 7. Implementing Type Class Instances

```haskell
-- Custom instance of Eq
data Point = Point Double Double

instance Eq Point where
  (Point x1 y1) == (Point x2 y2) = x1 == x2 && y1 == y2

-- Custom instance of Show
instance Show Point where
  show (Point x y) = "(" ++ show x ++ ", " ++ show y ++ ")"

-- Custom instance of Ord
instance Ord Point where
  compare (Point x1 y1) (Point x2 y2) = 
    compare (x1*x1 + y1*y1) (x2*x2 + y2*y2)  -- Compare by distance from origin

-- More complex: Functor for custom type
data Box a = Box a deriving (Show)

instance Functor Box where
  fmap f (Box a) = Box (f a)

-- Usage
fmap (*2) (Box 21)  -- Box 42
```

---

## 8. Practice Exercises

```haskell
-- Exercise 1: Create a Shape type with Circle and Rectangle
data Shape 
  = Circle Double           -- radius
  | Rectangle Double Double -- width, height
  deriving (Show)

area :: Shape -> Double
area (Circle r) = pi * r * r
area (Rectangle w h) = w * h

-- Exercise 2: Create a safe list indexing function
safeIndex :: [a] -> Int -> Maybe a
safeIndex [] _     = Nothing
safeIndex (x:_) 0  = Just x
safeIndex (_:xs) n
  | n < 0     = Nothing
  | otherwise = safeIndex xs (n - 1)

-- Exercise 3: Create a Binary Tree and implement depth function
data Tree a = Empty | Node a (Tree a) (Tree a)
  deriving (Show)

depth :: Tree a -> Int
depth Empty = 0
depth (Node _ left right) = 1 + max (depth left) (depth right)

-- Exercise 4: Implement Functor for Tree
instance Functor Tree where
  fmap _ Empty = Empty
  fmap f (Node x left right) = Node (f x) (fmap f left) (fmap f right)
```

---

## Next: [Part 3 - Functors, Applicatives, and Monads](./03-monads.md)
