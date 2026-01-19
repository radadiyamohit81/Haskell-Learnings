# Haskell Learning Path - Part 3: Functors, Applicatives, Monads & Effects

This is where Haskell gets powerful! These abstractions help handle effects, errors, and computations elegantly.

## Table of Contents

### Part A: Core Abstractions
1. [The Problem: Chaining Computations](#1-the-problem-chaining-computations)
2. [Functor: Mapping Over Wrapped Values](#2-functor-mapping-over-wrapped-values)
3. [Applicative: Functions in Boxes](#3-applicative-functions-in-boxes)
4. [Monad: Chaining Dependent Computations](#4-monad-chaining-dependent-computations)
5. [Do Notation: Syntactic Sugar for Monads](#5-do-notation-syntactic-sugar-for-monads)
6. [Common Monad Instances](#6-common-monad-instances)

### Part B: The IO Monad Deep Dive
7. [Understanding IO: The Gateway to the Real World](#7-understanding-io-the-gateway-to-the-real-world)
8. [IO Internals and Execution Model](#8-io-internals-and-execution-model)
9. [Common IO Operations](#9-common-io-operations)
10. [File Handling in IO](#10-file-handling-in-io)
11. [Exception Handling in IO](#11-exception-handling-in-io)
12. [Concurrency in IO](#12-concurrency-in-io)

### Part C: Building Custom Monads
13. [Creating Your Own Monad](#13-creating-your-own-monad)
14. [Monad Transformers](#14-monad-transformers)

### Part D: Real-World Application - FlowMonad (Vayu)
15. [The Monad Transformer Stack](#15-the-monad-transformer-stack)
16. [Application Startup Sequence](#16-application-startup-sequence)
17. [FlowMonad Internals](#17-flowmonad-internals)
18. [Request Lifecycle](#18-request-lifecycle)
19. [Concurrency Patterns in Flow](#19-concurrency-patterns-in-flow)
20. [Practical Examples](#20-practical-examples)

### Part E: Vayu Architecture - Three-Tier Service Pattern
21. [Architecture Overview](#21-architecture-overview)
22. [Layer 1: Generated API](#22-layer-1-generated-api-apihs)
23. [Layer 2: Routes Layer](#23-layer-2-routes-layer-corehs)
24. [Layer 3: Product Layer](#24-layer-3-product-layer)
25. [Layer 4: Internal Services](#25-layer-4-internal-services)
26. [Layer 5: External Services](#26-layer-5-external-services)
27. [Dependency Flow & Circular Prevention](#27-dependency-flow--circular-dependency-prevention)
28. [Complete Request Flow Example](#28-complete-request-flow-example-get-order)
29. [Key Architecture Principles](#29-key-architecture-principles)

### Part F: Practice & Summary
30. [Practice Exercises](#30-practice-exercises)
31. [Summary: The Complete Picture](#31-summary-the-complete-picture)

---

# Part A: Core Abstractions

---

## 1. The Problem: Chaining Computations

```haskell
-- Imagine we have these safe functions
safeHead :: [a] -> Maybe a
safeHead []    = Nothing
safeHead (x:_) = Just x

safeTail :: [a] -> Maybe [a]
safeTail []     = Nothing
safeTail (_:xs) = Just xs

-- Problem: How to chain them?
-- We want: get second element of a list

-- Ugly nested pattern matching
secondElement :: [a] -> Maybe a
secondElement xs = case safeTail xs of
  Nothing  -> Nothing
  Just xs' -> case safeHead xs' of
    Nothing -> Nothing
    Just x  -> Just x

-- This gets worse with more operations!
-- Functor, Applicative, and Monad solve this elegantly.
```

---

## 2. Functor: Mapping Over Wrapped Values

**Functor** = "I have a value in a box, apply a function to it without unwrapping"

### The Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│  Functor: Transform the VALUE inside, keep the CONTEXT     │
│                                                             │
│    fmap (+1)                                                │
│       │                                                     │
│       ▼                                                     │
│   ┌───────┐          ┌───────┐                             │
│   │ Just 5│   ──►    │ Just 6│                             │
│   └───────┘          └───────┘                             │
│     Maybe             Maybe                                 │
│    (context)         (context preserved)                    │
└─────────────────────────────────────────────────────────────┘
```

**Functor** = "I have a value in a box, apply a function to it without unwrapping"

### 2.1 The Functor Type Class

```haskell
class Functor f where
  fmap :: (a -> b) -> f a -> f b
  -- Also written as: (<$>) :: (a -> b) -> f a -> f b

-- Laws:
-- 1. Identity: fmap id == id
-- 2. Composition: fmap (f . g) == fmap f . fmap g
```

### 2.2 Functor Instances

```haskell
-- Maybe Functor
instance Functor Maybe where
  fmap _ Nothing  = Nothing
  fmap f (Just x) = Just (f x)

-- Usage
fmap (+1) (Just 5)   -- Just 6
fmap (+1) Nothing    -- Nothing
(+1) <$> Just 5      -- Just 6 (same as fmap)

-- List Functor
instance Functor [] where
  fmap = map

fmap (*2) [1, 2, 3]  -- [2, 4, 6]
(*2) <$> [1, 2, 3]   -- [2, 4, 6]

-- Either Functor (maps over Right)
instance Functor (Either e) where
  fmap _ (Left e)  = Left e
  fmap f (Right x) = Right (f x)

fmap (+1) (Right 5)           -- Right 6
fmap (+1) (Left "error")      -- Left "error"

-- Custom Functor
data Box a = Box a deriving (Show)

instance Functor Box where
  fmap f (Box a) = Box (f a)

fmap (*2) (Box 21)  -- Box 42
```

### 2.3 Functor in Practice

```haskell
-- Transform values inside containers
incrementAge :: Maybe Int -> Maybe Int
incrementAge = fmap (+1)

-- Parse and transform
parseAndDouble :: String -> Maybe Int
parseAndDouble s = fmap (*2) (readMaybe s)

-- Nested functors
-- fmap over Maybe inside List
users = [Just "Alice", Nothing, Just "Bob"]
upperUsers = fmap (fmap toUpper) users
-- [Just "ALICE", Nothing, Just "BOB"]
```

---

## 3. Applicative: Functions in Boxes

**Applicative** = "I have a function in a box AND a value in a box, apply them"

### The Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│  Applicative: Apply WRAPPED function to WRAPPED value      │
│                                                             │
│   ┌──────────┐    <*>    ┌───────┐     =    ┌───────┐     │
│   │ Just (+3)│           │ Just 5│          │ Just 8│     │
│   └──────────┘           └───────┘          └───────┘     │
│    Maybe (a->b)          Maybe a            Maybe b       │
│                                                             │
│  Key: Both function AND value are in the same context      │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 The Applicative Type Class

```haskell
class Functor f => Applicative f where
  pure  :: a -> f a                   -- Wrap value in minimal context
  (<*>) :: f (a -> b) -> f a -> f b   -- Apply wrapped function to wrapped value

-- Laws:
-- 1. Identity: pure id <*> v == v
-- 2. Composition: pure (.) <*> u <*> v <*> w == u <*> (v <*> w)
-- 3. Homomorphism: pure f <*> pure x == pure (f x)
-- 4. Interchange: u <*> pure y == pure ($ y) <*> u
```

### 3.2 Applicative Instances

```haskell
-- Maybe Applicative
instance Applicative Maybe where
  pure = Just
  Nothing <*> _  = Nothing
  _ <*> Nothing  = Nothing
  (Just f) <*> (Just x) = Just (f x)

-- Usage
pure 5 :: Maybe Int                    -- Just 5
Just (+3) <*> Just 5                   -- Just 8
Just (+) <*> Just 3 <*> Just 5         -- Just 8
Nothing <*> Just 5                     -- Nothing

-- List Applicative (all combinations)
instance Applicative [] where
  pure x = [x]
  fs <*> xs = [f x | f <- fs, x <- xs]

pure 5 :: [Int]                        -- [5]
[(+1), (*2)] <*> [1, 2, 3]             -- [2,3,4,2,4,6]
(+) <$> [1, 2] <*> [10, 20]            -- [11,21,12,22]
```

### 3.3 Applicative Style

```haskell
-- Pattern: pure f <*> x <*> y <*> z
-- Or:      f <$> x <*> y <*> z

-- Building data structures
data Person = Person String Int String
  deriving (Show)

-- Without Applicative (ugly)
createPerson :: Maybe String -> Maybe Int -> Maybe String -> Maybe Person
createPerson mName mAge mEmail = 
  case mName of
    Nothing -> Nothing
    Just name -> case mAge of
      Nothing -> Nothing
      Just age -> case mEmail of
        Nothing -> Nothing
        Just email -> Just (Person name age email)

-- With Applicative (clean!)
createPerson' :: Maybe String -> Maybe Int -> Maybe String -> Maybe Person
createPerson' mName mAge mEmail = 
  Person <$> mName <*> mAge <*> mEmail

-- Usage
createPerson' (Just "Mohit") (Just 25) (Just "mohit@example.com")
-- Just (Person "Mohit" 25 "mohit@example.com")

createPerson' (Just "Mohit") Nothing (Just "mohit@example.com")
-- Nothing (fails if any is Nothing)
```

### 3.4 Useful Applicative Functions

```haskell
-- liftA2: Apply binary function to two applicative values
liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
liftA2 f x y = f <$> x <*> y

liftA2 (+) (Just 3) (Just 5)  -- Just 8
liftA2 (,) (Just 3) (Just 5)  -- Just (3, 5)

-- sequenceA: Sequence applicative effects
sequenceA :: (Traversable t, Applicative f) => t (f a) -> f (t a)

sequenceA [Just 1, Just 2, Just 3]   -- Just [1, 2, 3]
sequenceA [Just 1, Nothing, Just 3]  -- Nothing

-- traverse: Map and sequence
traverse :: (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)

traverse safeHead [[1,2], [3,4], [5,6]]  -- Just [1, 3, 5]
traverse safeHead [[1,2], [], [5,6]]     -- Nothing
```

---

## 4. Monad: Chaining Dependent Computations

**Monad** = "I have a value in a box, apply a function that returns a boxed value, flatten the result"

The key difference from Applicative: **each step can depend on previous results**.

### The Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│  Monad: Chain computations where NEXT depends on PREVIOUS  │
│                                                             │
│   ┌───────┐     >>=     (a -> m b)      =    ┌───────┐    │
│   │ Just 5│             \x -> Just (x+1)     │ Just 6│    │
│   └───────┘                                  └───────┘    │
│     m a                                        m b        │
│                                                             │
│  Key: The function (a -> m b) can DECIDE what to return   │
│       based on the unwrapped value 'a'                    │
│                                                             │
│  Example of dependency:                                    │
│    Just 5 >>= \x -> if x > 3 then Just x else Nothing     │
│    The result DEPENDS on what 'x' is!                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 The Monad Type Class

```haskell
class Applicative m => Monad m where
  return :: a -> m a                    -- Same as 'pure'
  (>>=)  :: m a -> (a -> m b) -> m b    -- "bind" - chain computations
  (>>)   :: m a -> m b -> m b           -- Sequence, ignore first result
  
  -- Default implementations
  m >> n = m >>= \_ -> n

-- Laws:
-- 1. Left identity:  return a >>= f  ==  f a
-- 2. Right identity: m >>= return    ==  m
-- 3. Associativity:  (m >>= f) >>= g ==  m >>= (\x -> f x >>= g)
```

### 4.2 Understanding Bind (>>=)

```haskell
-- (>>=) :: m a -> (a -> m b) -> m b
-- Read as: "take value from box, apply function that returns boxed value"

-- Maybe Monad
instance Monad Maybe where
  return = Just
  Nothing >>= _ = Nothing
  (Just x) >>= f = f x

-- Example
Just 5 >>= \x -> Just (x + 1)          -- Just 6
Nothing >>= \x -> Just (x + 1)         -- Nothing
Just 5 >>= \x -> if x > 3 then Just x else Nothing  -- Just 5
Just 2 >>= \x -> if x > 3 then Just x else Nothing  -- Nothing
```

### 4.3 Chaining with Bind

```haskell
-- Remember our problem from the beginning?
secondElement :: [a] -> Maybe a
secondElement xs = 
  safeTail xs >>= \tail' ->   -- Get tail, if Nothing, stop
  safeHead tail'               -- Get head of tail

-- Chain multiple operations
thirdElement :: [a] -> Maybe a
thirdElement xs =
  safeTail xs >>= \t1 ->
  safeTail t1 >>= \t2 ->
  safeHead t2

-- More complex example
data User = User { userId :: Int, userName :: String } deriving (Show)
data Order = Order { orderId :: Int, orderUserId :: Int } deriving (Show)

getUser :: Int -> Maybe User
getUser 1 = Just (User 1 "Alice")
getUser 2 = Just (User 2 "Bob")
getUser _ = Nothing

getOrders :: Int -> Maybe [Order]
getOrders 1 = Just [Order 101 1, Order 102 1]
getOrders 2 = Just [Order 201 2]
getOrders _ = Nothing

-- Chain dependent lookups
getUserOrders :: Int -> Maybe (User, [Order])
getUserOrders uid =
  getUser uid >>= \user ->
  getOrders uid >>= \orders ->
  return (user, orders)

getUserOrders 1  -- Just (User 1 "Alice", [Order 101 1, Order 102 1])
getUserOrders 3  -- Nothing
```

---

## 5. Do Notation: Syntactic Sugar for Monads

**do notation** makes monadic code look imperative (but it's still functional!)

### 5.1 Translation Rules

```haskell
-- do notation is syntactic sugar for >>= and >>

-- Rule 1: do { x <- action; rest } == action >>= \x -> do { rest }
-- Rule 2: do { action; rest }      == action >> do { rest }
-- Rule 3: do { action }            == action

-- Example translation
-- do notation:
example = do
  x <- action1
  y <- action2
  action3
  return (x + y)

-- Desugared:
example' = 
  action1 >>= \x ->
  action2 >>= \y ->
  action3 >>
  return (x + y)
```

### 5.2 Do Notation Examples

```haskell
-- Our chain with do notation (MUCH cleaner!)
secondElement :: [a] -> Maybe a
secondElement xs = do
  tail' <- safeTail xs
  safeHead tail'

thirdElement :: [a] -> Maybe a
thirdElement xs = do
  t1 <- safeTail xs
  t2 <- safeTail t1
  safeHead t2

-- User orders example
getUserOrders :: Int -> Maybe (User, [Order])
getUserOrders uid = do
  user <- getUser uid
  orders <- getOrders uid
  return (user, orders)

-- With guards using 'guard' from Control.Monad
import Control.Monad (guard)

-- guard :: Alternative f => Bool -> f ()
-- guard True  = pure ()
-- guard False = empty  (Nothing for Maybe, [] for List)

adultUser :: Int -> Maybe User
adultUser uid = do
  user <- getUser uid
  age <- getUserAge uid
  guard (age >= 18)  -- Fails if not adult
  return user
```

### 5.3 Let in Do Blocks

```haskell
-- Use 'let' for pure computations in do blocks
processData :: Maybe Int -> Maybe Int -> Maybe String
processData mx my = do
  x <- mx
  y <- my
  let sum = x + y          -- Pure computation, no <-
      product = x * y
      message = "Sum: " ++ show sum ++ ", Product: " ++ show product
  return message

processData (Just 3) (Just 4)
-- Just "Sum: 7, Product: 12"
```

---

## 6. Common Monad Instances

### 6.1 List Monad (Non-determinism)

```haskell
instance Monad [] where
  return x = [x]
  xs >>= f = concatMap f xs  -- or [y | x <- xs, y <- f x]

-- List monad represents non-deterministic computation
-- Each >>= explores all possibilities

pairs :: [(Int, Int)]
pairs = do
  x <- [1, 2, 3]
  y <- [10, 20]
  return (x, y)
-- [(1,10),(1,20),(2,10),(2,20),(3,10),(3,20)]

-- With filtering
evenPairs :: [(Int, Int)]
evenPairs = do
  x <- [1, 2, 3]
  y <- [10, 20]
  guard (even x)
  return (x, y)
-- [(2,10),(2,20)]

-- Pythagorean triples
pythagTriples :: Int -> [(Int, Int, Int)]
pythagTriples n = do
  c <- [1..n]
  b <- [1..c]
  a <- [1..b]
  guard (a*a + b*b == c*c)
  return (a, b, c)
```

### 6.2 Either Monad (Error Handling)

```haskell
instance Monad (Either e) where
  return = Right
  Left e  >>= _ = Left e   -- Short-circuit on error
  Right x >>= f = f x

-- Error handling with Either
data AppError = NotFound | InvalidInput String | Unauthorized
  deriving (Show)

getUser :: Int -> Either AppError User
getUser 1 = Right (User 1 "Alice")
getUser _ = Left NotFound

validateAge :: Int -> Either AppError Int
validateAge age
  | age < 0   = Left (InvalidInput "Age cannot be negative")
  | age > 150 = Left (InvalidInput "Age too high")
  | otherwise = Right age

processUser :: Int -> Int -> Either AppError String
processUser uid age = do
  user <- getUser uid
  validAge <- validateAge age
  return $ userName user ++ " is " ++ show validAge ++ " years old"

processUser 1 25   -- Right "Alice is 25 years old"
processUser 2 25   -- Left NotFound
processUser 1 -5   -- Left (InvalidInput "Age cannot be negative")
```

---

# Part B: The IO Monad Deep Dive

---

## 7. Understanding IO: The Gateway to the Real World

### 7.1 Why IO Exists

Haskell is a **pure functional language** - functions cannot have side effects. But real programs need to:
- Read/write files
- Print to console
- Make network requests
- Get current time
- Generate random numbers

**The Problem**: How do you interact with the world in a pure language?

**The Solution**: The `IO` monad - a way to *describe* side effects without *performing* them.

### 7.2 What is IO?

```haskell
-- IO a = "A description of an action that, when executed, 
--         may perform side effects and produce a value of type 'a'"

-- Think of IO as a RECIPE, not the cooking itself
-- The recipe describes what to do, but doesn't do it until executed

-- Key insight: IO values are FIRST-CLASS
-- You can pass them around, store them, compose them - all pure!
recipe :: IO ()
recipe = putStrLn "Hello"   -- This doesn't print anything!
                             -- It's just a description of "print Hello"

-- The Haskell runtime executes 'main :: IO ()' - that's when effects happen
```

### 7.3 The IO Mental Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        THE IO TYPE                               │
│                                                                  │
│  Think of IO a as:                                              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    RECIPE / BLUEPRINT                    │   │
│  │                                                          │   │
│  │   "When executed, perform these effects and return 'a'"  │   │
│  │                                                          │   │
│  │   getLine :: IO String                                   │   │
│  │   = "Read a line from stdin, return the String"          │   │
│  │                                                          │   │
│  │   putStrLn "Hi" :: IO ()                                 │   │
│  │   = "Print 'Hi' to stdout, return nothing"               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  The recipe is PURE - creating it has no effects                │
│  Only the Haskell runtime can "cook" (execute) the recipe       │
└─────────────────────────────────────────────────────────────────┘
```

### 7.4 IO is NOT a Container

```haskell
-- WRONG mental model: IO String contains a String
-- RIGHT mental model: IO String is a computation that produces a String

-- You CANNOT extract a value from IO (safely)
-- This is intentional - it enforces sequencing and tracking of effects

-- Bad (doesn't exist):
-- unsafeExtract :: IO a -> a  -- NO! This would break purity

-- Good (use monadic operations):
main :: IO ()
main = do
  line <- getLine    -- "Bind" the result of IO String to 'line'
  putStrLn line      -- 'line' is a pure String here
```

### 7.5 The IO Monad Instance

```haskell
-- Simplified conceptual view (actual implementation is in GHC runtime)
instance Monad IO where
  return x = -- Create an IO action that does nothing and returns x
  
  (>>=) :: IO a -> (a -> IO b) -> IO b
  -- "First do this IO action, then pass its result to a function
  --  that returns another IO action, and do that"

-- Example:
greet :: IO ()
greet = getLine >>= \name -> putStrLn ("Hello, " ++ name)

-- Desugared do-notation:
greet' :: IO ()
greet' = do
  name <- getLine           -- Execute getLine, bind result to 'name'
  putStrLn ("Hello, " ++ name)  -- Execute putStrLn with that name
```

---

## 8. IO Internals and Execution Model

### 8.1 How IO Actually Works (Conceptually)

```haskell
-- IO can be thought of as:
type IO a = RealWorld -> (a, RealWorld)

-- Each IO action takes the "state of the world" and returns:
-- 1. A result value
-- 2. A new "state of the world"

-- This is why you can't run IO actions out of order - 
-- each one depends on the previous world state!

-- Conceptual execution:
-- main = do
--   x <- action1    -- world0 -> (x, world1)
--   y <- action2 x  -- world1 -> (y, world2)
--   action3 y       -- world2 -> ((), world3)
```

### 8.2 The Execution Model Visualized

```
┌─────────────────────────────────────────────────────────────────┐
│                     IO EXECUTION MODEL                           │
│                                                                  │
│   main :: IO ()                                                 │
│      │                                                           │
│      ▼                                                           │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                  │
│   │ World 0 │────►│ World 1 │────►│ World 2 │────► ...         │
│   └─────────┘     └─────────┘     └─────────┘                  │
│        │               │               │                         │
│        ▼               ▼               ▼                         │
│   getLine          process          putStrLn                    │
│   returns "Hi"     "Hi"             prints                      │
│                                                                  │
│   Key: The "world" is threaded through, ensuring ORDER          │
│        This is why IO operations are sequenced!                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Laziness and IO

```haskell
-- Haskell is lazy, but IO actions are executed in order!
-- This is because of the "world state" threading

main :: IO ()
main = do
  putStrLn "First"   -- Guaranteed to run first
  putStrLn "Second"  -- Guaranteed to run second
  putStrLn "Third"   -- Guaranteed to run third

-- However, PURE computations inside IO can be lazy:
main' :: IO ()
main' = do
  let x = expensiveComputation  -- NOT computed yet (lazy)
  putStrLn "Starting"
  print x                        -- NOW x is computed (forced by print)
```

### 8.4 IO Actions Are First-Class Values

```haskell
-- IO actions can be stored, passed around, and composed - all purely!

-- Store an action in a data structure
actions :: [IO ()]
actions = [putStrLn "One", putStrLn "Two", putStrLn "Three"]

-- Pass an action as an argument
runTwice :: IO a -> IO a
runTwice action = do
  action
  action

-- Return an action from a function
makeGreeter :: String -> IO ()
makeGreeter name = putStrLn ("Hello, " ++ name)

-- Compose actions
combined :: IO ()
combined = do
  sequence_ actions       -- Run all actions in the list
  runTwice (putStrLn "!")  -- Run the action twice
```

---

## 9. Common IO Operations

### 9.1 Console I/O

```haskell
-- OUTPUT

putStr :: String -> IO ()       -- Print string (no newline)
putStrLn :: String -> IO ()     -- Print string with newline
print :: Show a => a -> IO ()   -- Print any showable value

-- Examples:
putStr "Hello"      -- Prints: Hello (no newline)
putStrLn "World"    -- Prints: World\n
print 42            -- Prints: 42\n
print [1,2,3]       -- Prints: [1,2,3]\n

-- INPUT

getLine :: IO String            -- Read a line (without newline)
getChar :: IO Char              -- Read a single character
getContents :: IO String        -- Read ALL input (lazy)

-- Examples:
main :: IO ()
main = do
  putStr "Enter name: "
  name <- getLine
  putStrLn ("Hello, " ++ name)
```

### 9.2 Combining IO Actions

```haskell
import Control.Monad (when, unless, forever, forM, forM_, replicateM)

-- when: Conditional execution
when :: Applicative f => Bool -> f () -> f ()
main = do
  input <- getLine
  when (input == "quit") $ putStrLn "Goodbye!"

-- unless: Opposite of when
unless :: Applicative f => Bool -> f () -> f ()

-- forever: Run action infinitely
forever :: Applicative f => f a -> f b
main = forever $ do
  line <- getLine
  putStrLn $ "You said: " ++ line

-- forM: Map with effects (returns results)
forM :: (Traversable t, Monad m) => t a -> (a -> m b) -> m (t b)
results <- forM [1..5] $ \i -> do
  putStrLn $ "Processing " ++ show i
  return (i * 2)
-- results = [2,4,6,8,10]

-- forM_: Map with effects (discards results)
forM_ :: (Foldable t, Monad m) => t a -> (a -> m b) -> m ()
forM_ ["Alice", "Bob"] $ \name ->
  putStrLn $ "Hello, " ++ name

-- replicateM: Repeat action N times
replicateM :: Applicative m => Int -> m a -> m [a]
lines <- replicateM 3 getLine  -- Read 3 lines
```

### 9.3 Converting Between Pure and IO

```haskell
-- Pure value → IO value
return :: a -> IO a
pure :: a -> IO a  -- Same as return (Applicative)

example = do
  let x = 42           -- Pure computation
  y <- return (x + 1)  -- Wrap in IO (unnecessary but valid)
  print y

-- IO value → Pure value (ONLY via bind)
-- You CANNOT escape IO! This is by design.

extractExample :: IO ()
extractExample = do
  line <- getLine      -- line is now a pure String
  let upper = map toUpper line  -- Pure transformation
  putStrLn upper

-- Lifting pure functions to IO
fmap :: (a -> b) -> IO a -> IO b
(<$>) :: (a -> b) -> IO a -> IO b

-- Transform the result of an IO action
upperLine :: IO String
upperLine = map toUpper <$> getLine

-- Equivalent to:
upperLine' :: IO String
upperLine' = do
  line <- getLine
  return (map toUpper line)
```

---

## 10. File Handling in IO

### 10.1 Basic File Operations

```haskell
import System.IO

-- Reading files
readFile :: FilePath -> IO String      -- Read entire file (lazy)
readFile' :: FilePath -> IO String     -- Read entire file (strict, GHC 9.6+)

-- Writing files
writeFile :: FilePath -> String -> IO ()   -- Write (overwrite)
appendFile :: FilePath -> String -> IO ()  -- Append

-- Examples:
main :: IO ()
main = do
  -- Read
  contents <- readFile "input.txt"
  putStrLn contents
  
  -- Write
  writeFile "output.txt" "Hello, World!"
  
  -- Append
  appendFile "log.txt" "New log entry\n"
```

### 10.2 Handle-Based I/O

```haskell
import System.IO

-- Handles give more control than simple readFile/writeFile

-- Open modes
data IOMode = ReadMode | WriteMode | AppendMode | ReadWriteMode

-- Opening and closing
openFile :: FilePath -> IOMode -> IO Handle
hClose :: Handle -> IO ()

-- Reading from handles
hGetLine :: Handle -> IO String
hGetChar :: Handle -> IO Char
hGetContents :: Handle -> IO String

-- Writing to handles
hPutStr :: Handle -> String -> IO ()
hPutStrLn :: Handle -> String -> IO ()
hPrint :: Show a => Handle -> a -> IO ()

-- Example with handles:
processFile :: FilePath -> IO ()
processFile path = do
  handle <- openFile path ReadMode
  contents <- hGetContents handle
  putStrLn contents
  hClose handle
```

### 10.3 Safe Resource Handling with bracket

```haskell
import Control.Exception (bracket)

-- bracket ensures resources are released even if exceptions occur
bracket :: IO a        -- Acquire resource
        -> (a -> IO b) -- Release resource
        -> (a -> IO c) -- Use resource
        -> IO c

-- Safe file reading:
safeReadFile :: FilePath -> IO String
safeReadFile path = bracket
  (openFile path ReadMode)   -- Acquire: open file
  hClose                     -- Release: close file (always runs!)
  hGetContents               -- Use: read contents

-- Even better: withFile (built-in bracket for files)
withFile :: FilePath -> IOMode -> (Handle -> IO r) -> IO r

safeProcess :: FilePath -> IO ()
safeProcess path = withFile path ReadMode $ \handle -> do
  contents <- hGetContents handle
  putStrLn contents
  -- handle is automatically closed when this block exits
```

### 10.4 Binary File I/O

```haskell
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL

-- Strict ByteString (entire file in memory)
BS.readFile :: FilePath -> IO BS.ByteString
BS.writeFile :: FilePath -> BS.ByteString -> IO ()

-- Lazy ByteString (streamed, memory efficient for large files)
BL.readFile :: FilePath -> IO BL.ByteString
BL.writeFile :: FilePath -> BL.ByteString -> IO ()

-- Example: Copy a binary file
copyFile :: FilePath -> FilePath -> IO ()
copyFile src dst = do
  contents <- BL.readFile src
  BL.writeFile dst contents
```

---

## 11. Exception Handling in IO

### 11.1 The Exception Hierarchy

```haskell
import Control.Exception

-- Exceptions in Haskell form a hierarchy:
-- SomeException (root)
--   ├── IOException
--   ├── ArithException
--   │     ├── DivideByZero
--   │     └── Overflow
--   ├── ErrorCall
--   ├── PatternMatchFail
--   └── ... many more

-- You can catch specific exceptions or broader categories
```

### 11.2 Throwing Exceptions

```haskell
import Control.Exception

-- throw: Throw in pure code (avoid if possible!)
throw :: Exception e => e -> a

-- throwIO: Throw in IO (preferred)
throwIO :: Exception e => e -> IO a

-- error: Throw ErrorCall (for programmer errors)
error :: String -> a

-- Examples:
riskyFunction :: Int -> IO Int
riskyFunction n
  | n < 0     = throwIO $ userError "Negative input!"
  | otherwise = return (n * 2)
```

### 11.3 Catching Exceptions

```haskell
import Control.Exception

-- catch: Catch a specific exception type
catch :: Exception e => IO a -> (e -> IO a) -> IO a

-- try: Returns Either instead of throwing
try :: Exception e => IO a -> IO (Either e a)

-- catches: Catch multiple exception types
catches :: IO a -> [Handler a] -> IO a

-- Examples:

-- Using catch
safeRead :: FilePath -> IO String
safeRead path = readFile path `catch` \(e :: IOException) ->
  return $ "Error reading file: " ++ show e

-- Using try
safeRead' :: FilePath -> IO (Either IOException String)
safeRead' path = try (readFile path)

-- Using catches for multiple exception types
robustAction :: IO ()
robustAction = action `catches`
  [ Handler $ \(e :: IOException)   -> putStrLn $ "IO error: " ++ show e
  , Handler $ \(e :: ArithException) -> putStrLn $ "Math error: " ++ show e
  , Handler $ \(e :: SomeException)  -> putStrLn $ "Unknown: " ++ show e
  ]
```

### 11.4 Exception Handling Patterns

```haskell
import Control.Exception

-- finally: Cleanup that always runs
finally :: IO a -> IO b -> IO a
action `finally` cleanup

-- Example:
processWithCleanup :: IO ()
processWithCleanup = do
  putStrLn "Starting..."
  (riskyOperation `finally` putStrLn "Cleaning up...")
  putStrLn "Done"

-- onException: Cleanup only on exception
onException :: IO a -> IO b -> IO a

-- bracketOnError: Like bracket, but release only on error
bracketOnError :: IO a -> (a -> IO b) -> (a -> IO c) -> IO c

-- Complete error handling example:
robustFileProcessor :: FilePath -> IO ()
robustFileProcessor path = do
  result <- try (readFile path) :: IO (Either IOException String)
  case result of
    Left err -> do
      putStrLn $ "Failed to read file: " ++ show err
      putStrLn "Using default content..."
    Right contents -> do
      putStrLn $ "Read " ++ show (length contents) ++ " characters"
      processContents contents
```

---

## 12. Concurrency in IO

### 12.1 Basic Concurrency with forkIO

```haskell
import Control.Concurrent

-- forkIO: Spawn a new thread
forkIO :: IO () -> IO ThreadId

-- Basic example:
main :: IO ()
main = do
  forkIO $ putStrLn "Hello from thread!"
  putStrLn "Hello from main!"
  threadDelay 1000000  -- Wait 1 second (microseconds)

-- Thread communication with MVar
-- MVar a = "A mutable variable that can be empty or full"
newEmptyMVar :: IO (MVar a)        -- Create empty MVar
newMVar :: a -> IO (MVar a)         -- Create MVar with value
takeMVar :: MVar a -> IO a          -- Take value (blocks if empty)
putMVar :: MVar a -> a -> IO ()     -- Put value (blocks if full)
readMVar :: MVar a -> IO a          -- Read without taking
tryTakeMVar :: MVar a -> IO (Maybe a)  -- Non-blocking take
```

### 12.2 MVar Communication Pattern

```haskell
import Control.Concurrent

-- Producer-Consumer with MVar
producerConsumer :: IO ()
producerConsumer = do
  -- Create a channel (MVar)
  channel <- newEmptyMVar
  
  -- Producer thread
  forkIO $ do
    forM_ [1..5] $ \i -> do
      putStrLn $ "Producing: " ++ show i
      putMVar channel i
      threadDelay 500000
  
  -- Consumer (main thread)
  replicateM_ 5 $ do
    value <- takeMVar channel
    putStrLn $ "Consumed: " ++ show value
```

### 12.3 Software Transactional Memory (STM)

```haskell
import Control.Concurrent.STM

-- STM provides composable atomic transactions

-- TVar: Transactional variable
newTVar :: a -> STM (TVar a)
readTVar :: TVar a -> STM a
writeTVar :: TVar a -> a -> STM ()
modifyTVar :: TVar a -> (a -> a) -> STM ()

-- atomically: Run STM transaction in IO
atomically :: STM a -> IO a

-- Example: Thread-safe counter
type Counter = TVar Int

newCounter :: IO Counter
newCounter = atomically $ newTVar 0

increment :: Counter -> IO ()
increment counter = atomically $ modifyTVar counter (+1)

getCount :: Counter -> IO Int
getCount counter = atomically $ readTVar counter

-- Example with multiple variables (atomic transfer)
transfer :: TVar Int -> TVar Int -> Int -> STM ()
transfer from to amount = do
  fromBalance <- readTVar from
  when (fromBalance >= amount) $ do
    modifyTVar from (subtract amount)
    modifyTVar to (+ amount)

-- Run atomically
bankTransfer :: TVar Int -> TVar Int -> Int -> IO ()
bankTransfer from to amount = atomically $ transfer from to amount
```

### 12.4 Async Library (High-Level Concurrency)

```haskell
import Control.Concurrent.Async

-- async: Run action concurrently, get a handle
async :: IO a -> IO (Async a)

-- wait: Wait for result
wait :: Async a -> IO a

-- cancel: Cancel an async action
cancel :: Async a -> IO ()

-- concurrently: Run two actions concurrently, wait for both
concurrently :: IO a -> IO b -> IO (a, b)

-- race: Run two actions, return first to complete
race :: IO a -> IO b -> IO (Either a b)

-- mapConcurrently: Map over list concurrently
mapConcurrently :: Traversable t => (a -> IO b) -> t a -> IO (t b)

-- Examples:
fetchBothUrls :: IO (String, String)
fetchBothUrls = concurrently
  (fetchUrl "http://example.com")
  (fetchUrl "http://example.org")

fetchFirstResponse :: IO String
fetchFirstResponse = do
  result <- race
    (fetchUrl "http://server1.com")
    (fetchUrl "http://server2.com")
  return $ either id id result

processAllUrls :: [String] -> IO [String]
processAllUrls urls = mapConcurrently fetchUrl urls
```

### 12.5 Concurrency Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                    HASKELL CONCURRENCY                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      Main Thread                          │  │
│  │   main = do                                               │  │
│  │     forkIO task1  ─────────────────► [Task1 Thread]       │  │
│  │     forkIO task2  ─────────────────► [Task2 Thread]       │  │
│  │     ...                                    │      │       │  │
│  │     wait for results ◄─────────────────────┴──────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Communication:                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│  │  MVar   │    │  TVar   │    │ Channel │                    │
│  │ (1 val) │    │ (STM)   │    │ (queue) │                    │
│  └─────────┘    └─────────┘    └─────────┘                    │
│   Blocking     Composable       Unbounded                       │
│   Single val   Atomic txns      Multi-val                       │
└─────────────────────────────────────────────────────────────────┘
```

---

# Part C: Building Custom Monads

---

## 13. Creating Your Own Monad

### 13.1 State Monad (Manual Implementation)

```haskell
-- State monad: computations that read/modify state

-- newtype State s a = State { runState :: s -> (a, s) }
-- "A computation that takes state, returns value and new state"

newtype State s a = State { runState :: s -> (a, s) }

instance Functor (State s) where
  fmap f (State run) = State $ \s ->
    let (a, s') = run s
    in (f a, s')

instance Applicative (State s) where
  pure a = State $ \s -> (a, s)
  (State sf) <*> (State sa) = State $ \s ->
    let (f, s')  = sf s
        (a, s'') = sa s'
    in (f a, s'')

instance Monad (State s) where
  return = pure
  (State run) >>= f = State $ \s ->
    let (a, s') = run s
        (State run') = f a
    in run' s'

-- Helper functions
get :: State s s
get = State $ \s -> (s, s)

put :: s -> State s ()
put s = State $ \_ -> ((), s)

modify :: (s -> s) -> State s ()
modify f = State $ \s -> ((), f s)

-- Example: Stack operations
type Stack = [Int]

pop :: State Stack Int
pop = do
  stack <- get
  case stack of
    []     -> error "Empty stack"
    (x:xs) -> do
      put xs
      return x

push :: Int -> State Stack ()
push x = modify (x:)

stackOps :: State Stack Int
stackOps = do
  push 1
  push 2
  push 3
  a <- pop
  b <- pop
  return (a + b)

-- Run it
runState stackOps []  -- (5, [1])
```

### 13.2 Reader Monad (Environment/Config)

```haskell
-- Reader monad: computations that read from shared environment

newtype Reader r a = Reader { runReader :: r -> a }

instance Functor (Reader r) where
  fmap f (Reader run) = Reader $ f . run

instance Applicative (Reader r) where
  pure a = Reader $ const a
  (Reader rf) <*> (Reader ra) = Reader $ \r -> rf r (ra r)

instance Monad (Reader r) where
  return = pure
  (Reader run) >>= f = Reader $ \r ->
    let a = run r
        (Reader run') = f a
    in run' r

ask :: Reader r r
ask = Reader id

asks :: (r -> a) -> Reader r a
asks f = Reader f

-- Example: Config-based computation
data Config = Config
  { configHost :: String
  , configPort :: Int
  , configDebug :: Bool
  }

type App = Reader Config

getConnectionString :: App String
getConnectionString = do
  config <- ask
  return $ configHost config ++ ":" ++ show (configPort config)

logIfDebug :: String -> App ()
logIfDebug msg = do
  debug <- asks configDebug
  if debug
    then return ()  -- In real code: would log
    else return ()

-- Run it
config = Config "localhost" 5432 True
runReader getConnectionString config  -- "localhost:5432"
```

---

## 14. Monad Transformers

When you need to combine multiple effects:

```haskell
-- Problem: We want Maybe AND State

-- Solution: Monad Transformers
import Control.Monad.Trans.State
import Control.Monad.Trans.Maybe
import Control.Monad.Trans.Class (lift)

-- MaybeT wraps Maybe inside another monad
-- MaybeT (State s) a = computation with state that might fail

type App s = MaybeT (State s)

-- Example combining State and Maybe
safePop :: App [Int] Int
safePop = do
  stack <- lift get
  case stack of
    []     -> MaybeT (return Nothing)  -- Fail
    (x:xs) -> do
      lift (put xs)
      return x
```

---

# Part D: Real-World Application - FlowMonad (Vayu)

This section covers the production monad stack used in Vayu, demonstrating how all these concepts come together in a real application.

---

## 15. The Monad Transformer Stack

### 15.1 What is a Monad Transformer?

A monad transformer wraps one monad inside another, combining their capabilities:

```haskell
-- ReaderT adds "read-only environment" capability to any monad
newtype ReaderT r m a = ReaderT { runReaderT :: r -> m a }

-- ExceptT adds "error handling" capability
newtype ExceptT e m a = ExceptT { runExceptT :: m (Either e a) }
```

### 15.2 Vayu's Monad Stack (Bottom to Top)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Flow                                     │
│  type Flow = ReaderT Env (EulerHS.Flow)                         │
│                                                                  │
│  Provides: Access to Env (config, sessionId, etc.)              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EulerHS.Flow                                  │
│  (Free monad over FlowMethod functor)                           │
│                                                                  │
│  Provides: DB, Redis, HTTP, Logging, Forking                    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         IO                                       │
│  (Base monad - actual side effects)                             │
│                                                                  │
│  Provides: Real-world interactions                              │
└─────────────────────────────────────────────────────────────────┘
```

### 15.3 The Flow Type Definition

```haskell
-- In FlowMonad.hs (line ~293)
type Flow = EulerLanguage.ReaderFlow Env

-- EulerLanguage.ReaderFlow is defined as:
type ReaderFlow env = ReaderT env Flow

-- So expanded:
type Flow = ReaderT Env (EulerHS.Language.Flow)
```

This means every `Flow` computation:
1. Has access to an `Env` value (via ReaderT)
2. Can perform database, Redis, HTTP operations (via EulerHS.Flow)
3. Eventually runs in IO

### 15.4 Common Monads Summary Table

| Monad | Purpose | Context |
|-------|---------|---------|
| `IO` | Side effects | Real-world interaction |
| `Maybe` | Optional values | Failure/absence |
| `Either e` | Error handling | Failure with info |
| `Reader r` | Implicit environment | Dependency injection |
| `State s` | Mutable state | Stateful computation |
| `[]` (List) | Non-determinism | Multiple results |

---

## 16. Application Startup Sequence

### 16.1 Entry Point: `main`

```haskell
-- app/Main.hs
module Main (main) where

import Vayu.App (runVayuBackendApp)

main :: IO ()
main = runVayuBackendApp
```

### 16.2 Detailed Startup: `runVayuBackendApp`

```haskell
-- src/Vayu/App.hs
runVayuBackendApp :: IO ()
runVayuBackendApp = do
  -- ┌─────────────────────────────────────────────┐
  -- │ PHASE 1: Environment Detection              │
  -- └─────────────────────────────────────────────┘
  appEnvType <- Config.getEnv . maybe "PROD" ByteStringUTF.fromString 
                <$> Environment.lookupEnv "VAYU_APP_ENV"
  
  -- ┌─────────────────────────────────────────────┐
  -- │ PHASE 2: Signal Handling Setup              │
  -- └─────────────────────────────────────────────┘
  sigTermMVar <- ConcurrencyControl.newEmptyMVar
  void $ Signals.installHandler Signals.sigTERM 
         (Signals.Catch $ ProcessTrackerUtils.onSigTerm sigTermMVar) Nothing

  -- ┌─────────────────────────────────────────────┐
  -- │ PHASE 3: Logger Configuration               │
  -- └─────────────────────────────────────────────┘
  let loggerCfg = EulerTypes.LoggerConfig
        { EulerTypes._isAsync = True
        , EulerTypes._logLevel = EulerTypes.Debug
        , EulerTypes._logFilePath = "/tmp/vayu-hs.log"
        , EulerTypes._logToConsole = True
        , EulerTypes._logToFile = False
        , EulerTypes._maxQueueSize = 1000
        , EulerTypes._logRawSql = EulerTypes.SafelyOmitSqlLogs
        , EulerTypes._logMaskingConfig = Just loggingMask
        , ...
        }

  -- ┌─────────────────────────────────────────────┐
  -- │ PHASE 4: FlowRuntime Initialization         │
  -- └─────────────────────────────────────────────┘
  EulerRuntime.withFlowRuntime (Just createLoggerRuntime) $ \flowRt' -> do
    
    -- ┌─────────────────────────────────────────────┐
    -- │ PHASE 5: Configuration Loading              │
    -- └─────────────────────────────────────────────┘
    configs <- EulerInterpreter.runFlow flowRt' Config.config'
    
    -- ┌─────────────────────────────────────────────┐
    -- │ PHASE 6: Database & Redis Connection        │
    -- └─────────────────────────────────────────────┘
    try (EulerInterpreter.runFlow flowRt' 
          (DatabaseConfig.prepareDBConnections configs
           *> RedisConfig.prepareRedisPubSubConnection redisMode configs))
      >>= \case
        Left (e :: SomeException) -> putStrLn ("Exception: " ++ show e)
        Right pubSubConnection -> do
          
          -- ┌─────────────────────────────────────────────┐
          -- │ PHASE 7: HTTP Manager Setup                 │
          -- └─────────────────────────────────────────────┘
          manager <- Client.newManager mSetting
          managerProxy <- Client.newManager mProxySetting
          
          let flowRt = flowRt'
                { EulerRuntime._httpClientManagers = proxyMap
                , EulerRuntime._defaultHttpClientManager = manager
                , EulerRuntime._pubSubConnection = pubSubConnection
                }
          
          -- ┌─────────────────────────────────────────────┐
          -- │ PHASE 8: Environment (Env) Construction     │
          -- └─────────────────────────────────────────────┘
          randomSessionUUID <- UUID.toText <$> UUIDV4.nextRandom
          randomRequestUUID <- UUID.toText <$> UUIDV4.nextRandom
          emptyLocalCache <- Cache.newCacheIO Nothing
          
          let env = AppTypes.Env 
                flowRt           -- Runtime with all connections
                configs          -- Application configuration
                randomSessionUUID
                randomRequestUUID
                ""               -- shopUrl (empty at startup)
                randomDeviceUUID
                emptyLocalCache
                False            -- testingMode
                Nothing          -- kafkaRuntime (set later)
                sigTermMVar      -- shutdown signal
          
          -- ┌─────────────────────────────────────────────┐
          -- │ PHASE 9: Background Services                │
          -- └─────────────────────────────────────────────┘
          void $ EulerInterpreter.runFlow flowRt $ 
                 runReaderT Cache.subscribeToCacheInvalidationChannel env
          
          -- ┌─────────────────────────────────────────────┐
          -- │ PHASE 10: Start HTTP Server                 │
          -- └─────────────────────────────────────────────┘
          bracket
            (KafkaProducer.initKafka env)
            (\e -> KafkaProducer.closeKafkaRuntime (kafkaRuntime e))
            $ \env' -> runSettings settings =<< runVayuApp env'
```

### 16.3 Visual Startup Timeline

```
Time ──────────────────────────────────────────────────────────────►

│ main()
│   │
│   ▼
│ runVayuBackendApp
│   │
│   ├──► Read VAYU_APP_ENV
│   ├──► Install SIGTERM handler
│   ├──► Configure logger
│   │
│   ├──► withFlowRuntime ─────────────────────────────────────────┐
│   │       │                                                      │
│   │       ├──► Create LoggerRuntime                             │
│   │       ├──► Create CoreRuntime                               │
│   │       ├──► Initialize FlowRuntime                           │
│   │       │                                                      │
│   │       ├──► runFlow: Config.config' ◄── Load YAML + ENV      │
│   │       │                                                      │
│   │       ├──► runFlow: prepareDBConnections                    │
│   │       │       ├── Create connection pools                   │
│   │       │       └── Verify connectivity                       │
│   │       │                                                      │
│   │       ├──► runFlow: prepareRedisPubSubConnection            │
│   │       │       ├── Connect to Redis cluster/standalone       │
│   │       │       └── Setup pub/sub                             │
│   │       │                                                      │
│   │       ├──► Setup HTTP managers (proxy, low-latency)         │
│   │       │                                                      │
│   │       ├──► Construct Env                                    │
│   │       │                                                      │
│   │       ├──► Subscribe to cache invalidation                  │
│   │       │                                                      │
│   │       ├──► Initialize Kafka producer                        │
│   │       │                                                      │
│   │       └──► Warp.runSettings ◄── Server listening!           │
│   │               │                                              │
│   │               └── (Server runs until shutdown)              │
│   │                                                              │
│   └──────────────────────────────────────────────────────────────┘
```

---

## 17. FlowMonad Internals

### 17.1 The Env Type

```haskell
-- src/Vayu/Types/App.hs
data Env = Env
    { runTime        :: !(EulerRuntime.FlowRuntime Text)
      -- ^ Contains:
      --   • Database connection pools
      --   • Redis connections
      --   • HTTP client managers
      --   • Logger runtime
      --   • Pub/Sub connection
      
    , config         :: Config.Config
      -- ^ Application configuration:
      --   • Feature flags
      --   • API endpoints
      --   • TTLs and limits
      --   • Shop-specific settings
      
    , sessionId      :: Text
      -- ^ Unique session identifier (per server instance)
      
    , requestId      :: Text
      -- ^ Unique request identifier (per HTTP request)
      
    , shopUrl        :: Text
      -- ^ Current shop context (set per request)
      
    , deviceId       :: Text
      -- ^ Device identifier from request
      
    , localCache     :: CacheTypes.LocalCache
      -- ^ In-memory caches:
      --   • Merchant cache
      --   • Shop cache
      --   • High-risk pincode cache
      
    , testingMode    :: Bool
      -- ^ Flag for test execution
      
    , kafkaRuntime   :: Maybe KafkaTypes.KafkaRuntime
      -- ^ Kafka producer for async logging
      
    , shutdownSignal :: MVar ()
      -- ^ Signal for graceful shutdown
    }
```

### 17.2 How `ask` Works in Flow

The `ask` function comes from the `MonadReader` typeclass:

```haskell
class Monad m => MonadReader r m | m -> r where
  ask :: m r              -- Get the entire environment
  local :: (r -> r) -> m a -> m a  -- Modify env locally

-- For ReaderT:
instance Monad m => MonadReader r (ReaderT r m) where
  ask = ReaderT return
  -- Equivalent to: ask = ReaderT (\env -> return env)
```

When you write:

```haskell
getConfig :: Flow Config.Config
getConfig = do
  Env {..} <- ask   -- Pattern match on Env
  return config     -- Extract config field
```

It desugars to:

```haskell
getConfig :: Flow Config.Config
getConfig = ask >>= \(Env {..}) -> return config
```

### 17.3 The `lift` Function

To access EulerHS capabilities from Flow, we use `lift`:

```haskell
-- lift takes an action from the inner monad and "lifts" it
lift :: (MonadTrans t, Monad m) => m a -> t m a

-- In our stack:
-- Flow = ReaderT Env (EulerHS.Flow)
-- So: lift :: EulerHS.Flow a -> Flow a

-- Example: Running IO in Flow
runIO :: IO a -> Flow a
runIO action = lift $ EulerLanguage.runIO action
```

### 17.4 Composing Flow Actions

```haskell
-- Sequential composition with do-notation
myHandler :: Text -> Flow Response
myHandler shopId = do
  -- Each line is a Flow action
  config <- getConfig                    -- Flow Config
  shop <- ShopQueries.findById shopId    -- Flow (Either Error Shop)
  
  case shop of
    Left err -> throwError err
    Right s -> do
      orders <- OrderQueries.findByShop (s ^. id)  -- Flow [Order]
      return $ Response s orders

-- This desugars to:
myHandler shopId =
  getConfig >>= \config ->
  ShopQueries.findById shopId >>= \shop ->
  case shop of
    Left err -> throwError err
    Right s -> 
      OrderQueries.findByShop (s ^. id) >>= \orders ->
      return $ Response s orders
```

---

## 18. Request Lifecycle

### 18.1 From HTTP Request to Flow Execution

```
┌─────────────────────────────────────────────────────────────────┐
│                     HTTP Request Arrives                         │
│                   (Warp receives request)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Middlewares                                │
│  • Cors.vayuCors                                                │
│  • Response.responseMiddleware                                  │
│  • Request.requestMiddlewares                                   │
│  • Request.headersMiddleware                                    │
│  • Request.webhookEmptyBodyMiddleware                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Servant Routing                             │
│  API.vayuAPIProxy matches request to handler                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     hoistServer                                  │
│                                                                  │
│  runVayu :: Env -> Server API.VayuAPI                           │
│  runVayu env = hoistServer API.vayuAPIProxy (f env) API.runVayu'│
│                                                                  │
│  f :: Env -> ReaderT Env (ExceptT ServerError IO) a -> Handler a│
│  f env r = do                                                   │
│    eResult <- liftIO $ runExceptT $ runReaderT r env            │
│    case eResult of                                              │
│      Left err -> throwError err                                 │
│      Right res -> pure res                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Your Handler Executes                         │
│                                                                  │
│  myEndpoint :: Text -> Flow Response                            │
│  myEndpoint shopId = do                                         │
│    config <- getConfig      -- Reads from Env                   │
│    shop <- findShop shopId  -- DB query via EulerHS             │
│    ...                                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Response Serialization                         │
│  Handler returns, Servant serializes to JSON                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    HTTP Response Sent                            │
└─────────────────────────────────────────────────────────────────┘
```

### 18.2 The `hoistServer` Transformation

`hoistServer` transforms one monad into another for all handlers:

```haskell
-- Servant's hoistServer signature
hoistServer :: HasServer api '[] 
            => Proxy api 
            -> (forall x. m x -> n x)  -- Natural transformation
            -> ServerT api m 
            -> ServerT api n

-- In Vayu:
hoistServer API.vayuAPIProxy (f env) API.runVayu'

-- Where f transforms:
--   ReaderT Env (ExceptT ServerError IO) a  →  Handler a

-- By running:
--   1. runReaderT with env   : ReaderT Env m a → m a
--   2. runExceptT            : ExceptT e m a → m (Either e a)
--   3. Pattern match Either  : Either ServerError a → Handler a
```

---

## 19. Concurrency Patterns in Flow

### 19.1 `spawnThread` - Fire and Forget

```haskell
spawnThread :: Flow a -> Flow ()
spawnThread runner = do
  appEnv <- ask                      -- Get current environment
  void $ spawnThreadAwait runner appEnv  -- Fork with same env

-- Usage:
myHandler = do
  result <- computeResult
  spawnThread $ logToAnalytics result  -- Fire and forget
  return result
```

### 19.2 `spawnThreadAwait` - Fork with Result

```haskell
spawnThreadAwait :: Flow a -> Env -> Flow (Awaitable (Either Text a))
spawnThreadAwait runner env =
  lift $                             -- Lift to EulerHS.Flow
    forkFlow' "Forking Flow" $       -- Fork in EulerHS
      runReaderT runner env          -- Run with same environment

-- Usage:
myHandler = do
  awaitable <- spawnThreadAwait' "fetch-data" fetchData
  -- ... do other work ...
  result <- await Nothing awaitable  -- Block until result
  return result
```

### 19.3 `parallely` - Concurrent Execution

```haskell
parallely :: [Flow a] -> Flow [a]
parallely flows = do
  -- Fork all flows
  awaitables <- mapM (spawnThreadAwait' "parallely") flows
  -- Wait for all results
  results <- mapM (await Nothing) awaitables
  -- Return only successes
  return $ rights results

-- Usage:
myHandler shopIds = do
  -- Fetch all shops concurrently
  shops <- parallely $ map ShopQueries.findById shopIds
  return shops
```

### 19.4 `forkFlowParallel` - Two Concurrent Flows

```haskell
forkFlowParallel :: Flow a -> Flow b -> Flow (Either Text (a, b))
forkFlowParallel flowA flowB = do
  awaitableA <- spawnThreadAwait' "forkFlowParallel_A" flowA
  awaitableB <- spawnThreadAwait' "forkFlowParallel_B" flowB

  resultA <- await Nothing awaitableA
  resultB <- await Nothing awaitableB

  case (resultA, resultB) of
    (Right a, Right b) -> return $ Right (a, b)
    (Left errA, _)     -> return $ Left errA
    (_, Left errB)     -> return $ Left errB

-- Usage:
myHandler = do
  result <- forkFlowParallel 
              (fetchUserData userId)
              (fetchOrderHistory userId)
  case result of
    Right (user, orders) -> processData user orders
    Left err -> handleError err
```

### 19.5 Visual: Thread Execution

```
Main Flow Thread
     │
     ├──► spawnThread (analytics)  ──────────────────────►  [Analytics Thread]
     │                                                              │
     ├──► Continue execution                                       │
     │                                                              │
     ├──► spawnThreadAwait (fetch)  ──────────────────────►  [Fetch Thread]
     │         │                                                    │
     │         │◄──────── await ─────────────────────────────────────
     │         │
     │         ▼
     ├──► Use fetched result
     │
     └──► Return response
```

---

## 20. Practical Examples

### 20.1 Reading Configuration

```haskell
-- Simple config read
getApiVersion :: Flow String
getApiVersion = do
  config <- getConfig
  return $ config ^. Accessor.shopifyApiVersion

-- Config read with computation
getShopifyApiVersionForShop :: Text -> Flow String
getShopifyApiVersionForShop shopUrl = do
  config <- getConfig
  let shopList = config ^. Accessor.shopUrlListForNewShopifyApiVersion
  let version = if shopUrl `elem` shopList
                then config ^. Accessor.newShopifyApiVersion
                else Text.unpack (config ^. Accessor.shopifyApiVersion)
  return $ "/admin/api/" <> version
```

### 20.2 Database Query

```haskell
-- In src/Vayu/Services/Internal/*/Queries.hs
findOrderById :: Text -> Flow (Either Error Order)
findOrderById orderId = do
  schemaName <- getSchemaName "public"
  VayuQueries.findOneEither tag (dbTable schemaName) (predicate filterBy)
    >>= DBError.handleDBResponse errMsg DBError.Find
  where
    filterBy = FilterById orderId
    tag = "query:findOrderById"
    errMsg = "Unable to find order"
```

### 20.3 HTTP Call with Logging

```haskell
fetchExternalData :: Text -> Flow (Either Error ExternalData)
fetchExternalData entityId = do
  -- Get configuration
  baseUrl <- getBaseUrlFromConfig
  sessionId <- getSessionId
  
  -- Log the attempt
  Logger.logInfo tag ["entityId" .= entityId, "sessionId" .= sessionId]
  
  -- Make HTTP call (via EulerHS)
  response <- callServantAPI (mkClient baseUrl) (getEntity entityId)
  
  -- Handle response
  case response of
    Left err -> do
      Logger.logFailure tag ("External call failed: " <> show err)
      return $ Left $ mkError err
    Right data' -> do
      Logger.logInfo tag ["status" .= "success"]
      return $ Right data'
  where
    tag = "External:fetchExternalData"
```

### 20.4 Transaction with Multiple Operations

```haskell
processPayment :: PaymentRequest -> Flow (Either Error PaymentResponse)
processPayment request = do
  -- Start transaction context
  sessionId <- getSessionId
  
  -- Step 1: Validate
  validationResult <- validatePayment request
  case validationResult of
    Left err -> return $ Left err
    Right validated -> do
      
      -- Step 2: Create order on platform (parallel with inventory check)
      (orderResult, inventoryResult) <- forkFlowParallel
        (PlatformOrder.create validated)
        (Inventory.checkAvailability validated)
      
      case (orderResult, inventoryResult) of
        (Left err, _) -> return $ Left err
        (_, Left err) -> return $ Left err
        (Right order, Right inventory) -> do
          
          -- Step 3: Process payment
          paymentResult <- PaymentGateway.charge order
          
          -- Step 4: Fire async events
          spawnThread $ do
            Analytics.track "payment_processed" order
            Notification.send order
          
          return paymentResult
```

### 20.5 Error Handling Pattern

```haskell
-- Using Either for error handling
createOrder :: OrderRequest -> Flow (Either Error Order)
createOrder request = do
  -- Validate request
  case validateRequest request of
    Left validationErr -> return $ Left validationErr
    Right validReq -> do
      -- Try to create
      dbResult <- OrderQueries.create validReq
      case dbResult of
        Left dbErr -> do
          Logger.logFailure tag (show dbErr)
          return $ Left dbErr
        Right order -> do
          Logger.logInfo tag ["orderId" .= (order ^. id)]
          return $ Right order
  where
    tag = "Order:create"

-- In Product layer - handle and throw HTTP error
handleCreateOrder :: OrderRequest -> Flow Order
handleCreateOrder request = do
  result <- createOrder request
  case result of
    Left err -> Logger.logAndThrow400 "handleCreateOrder" err
    Right order -> return order
```

---

# Part E: Vayu Architecture - Three-Tier Service Pattern

---

## 21. Architecture Overview

Vayu follows a strict **three-tier architecture** for handling HTTP requests:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HTTP Request                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    1. Generated API Layer                                │
│                    (src/Vayu/Generated/API.hs)                          │
│                                                                          │
│  • Auto-generated from OpenAPI spec (doc/Api.yaml)                      │
│  • Defines VayuAPI type with all endpoints                              │
│  • Maps routes to CoreRoutes handlers                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    2. Routes Layer (Core.hs)                             │
│                    (src/Vayu/Routes/Core.hs)                            │
│                                                                          │
│  • Thin routing layer - minimal logic                                   │
│  • Authentication middleware application                                │
│  • Request/Response transformation                                      │
│  • Delegates to Product layer                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    3. Product Layer                                      │
│                    (src/Vayu/Product/*)                                  │
│                                                                          │
│  • Business logic orchestration                                         │
│  • Workflow composition                                                 │
│  • Calls Internal + External Services                                   │
│  • Error handling at API boundary                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│   4. Internal Services      │   │   5. External Services      │
│   (Services/Internal/*)     │   │   (Services/External/*)     │
│                             │   │                             │
│ • Database operations       │   │ • Shopify API calls         │
│ • Pure business functions   │   │ • WooCommerce integration   │
│ • Reusable logic            │   │ • Payment gateways (Euler)  │
│ • Redis operations          │   │ • Notification providers    │
└─────────────────────────────┘   └─────────────────────────────┘
```

---

## 22. Layer 1: Generated API (API.hs)

**Location**: `src/Vayu/Generated/API.hs`

This file is **auto-generated** from the OpenAPI specification and should **never be edited manually**.

```haskell
-- src/Vayu/Generated/API.hs (auto-generated)
module Vayu.Generated.API
  ( VayuBackend (..)
  , VayuAPI
  , vayuAPIProxy
  , runVayu'
  ) where

import Vayu.Routes.Core as CoreRoutes

-- Type-level API definition (hundreds of endpoints)
type VayuAPI = 
       "abandon" :> "checkout" :> ... :> Post '[JSON] ResponseWrapper
  :<|> "address" :> ... :> Get '[JSON] AddressResponse
  :<|> "order" :> Capture "orderId" Text :> Get '[JSON] GetOrderResponse
  -- ... many more endpoints

-- Maps each endpoint to its handler in CoreRoutes
runVayu' :: ServerT VayuAPI (ReaderT Env (ExceptT ServerError IO))
runVayu' = 
       CoreRoutes.abandonCheckout
  :<|> CoreRoutes.getAddresses
  :<|> CoreRoutes.getOrderFromId
  -- ... all handlers mapped
```

**Key Points**:
- Generated by `./scripts/generate-backend.sh`
- Defines the `VayuAPI` servant type
- Maps each route to a handler in `CoreRoutes`
- Regenerate after modifying `doc/Api.yaml`

---

## 23. Layer 2: Routes Layer (Core.hs)

**Location**: `src/Vayu/Routes/Core.hs`

The routing layer is a **thin orchestration layer** that:
1. Applies authentication middleware
2. Extracts request context (sessionId, shopUrl, etc.)
3. Delegates to Product layer
4. Handles HTTP-specific concerns

```haskell
-- src/Vayu/Routes/Core.hs
module Vayu.Routes.Core (
  getOrderFromId,
  getStartPayment,
  sessionLogin,
  -- ... 200+ route handlers
) where

import qualified Vayu.Product.Order.Main as Order
import qualified Vayu.Product.Identity.Main as Identity
import qualified Vayu.Middlewares.Authentication as AuthMiddleware

-- Route handler pattern: Extract context → Apply auth → Call Product layer
getOrderFromId ::
  OrderId ->
  Maybe IsSyncOrderFlow ->
  Maybe AccessToken ->
  Maybe SessionId ->
  Maybe RequestId ->
  Maybe ShopUrl ->
  Maybe TestingMode ->
  Maybe DeviceId ->
  FlowHandler GenTypes.GetOrderResponse
getOrderFromId orderId isSyncOrderFlow accessToken sessionId requestId shopUrl testingMode deviceId =
  flowWithTrace  -- Sets up Flow context with trace info
    ( AuthMiddleware.validateS2SAuth accessToken READ ORDER
        >>= UtilsCommon.handleBoolWithException
        >> Order.getOrderWithId orderId (fromMaybe False isSyncOrderFlow)
    )
    sessionId requestId shopUrl testingMode deviceId

-- Auth + Token validation pattern
sessionLogin ::
  GenTypes.LoginRequest ->
  Maybe SessionId ->
  Maybe RequestId ->
  Maybe ShopUrl ->
  Maybe TestingMode ->
  Maybe DeviceId ->
  FlowHandler GenTypes.LoginResponse
sessionLogin reqBody sessionId requestId shopURL testingMode deviceId =
  flowWithTrace
    (Identity.login reqBody shopURL)  -- Direct call to Product layer
    sessionId requestId shopURL testingMode deviceId
```

### The `flowWithTrace` Helper

```haskell
flowWithTrace ::
  Flow a ->
  Maybe SessionId ->
  Maybe RequestId ->
  Maybe ShopUrl ->
  Maybe TestingMode ->
  Maybe DeviceId ->
  FlowHandler a
flowWithTrace flow mSessionId mRequestId mShopUrl mTestingMode mDeviceId = do
  -- Build enriched environment with request context
  env <- ask
  let env' = env 
        { sessionId = fromMaybe (sessionId env) mSessionId
        , requestId = fromMaybe (requestId env) mRequestId
        , shopUrl = fromMaybe "" mShopUrl
        , testingMode = fromMaybe False mTestingMode
        , deviceId = fromMaybe "" mDeviceId
        }
  -- Run the Flow with enriched context
  liftIO $ EulerInterpreter.runFlow (runTime env') $ runReaderT flow env'
```

---

## 24. Layer 3: Product Layer

**Location**: `src/Vayu/Product/*/Main.hs`

The Product layer contains **business logic workflows** that orchestrate multiple services.

```haskell
-- src/Vayu/Product/Order/Main.hs
module Vayu.Product.Order.Main (
  getOrderWithId,
  retryOrderFromId,
  updateOrder,
  -- ... business operations
) where

-- Import Internal Services (for DB/business logic)
import qualified Vayu.Services.Internal.Order.Main as OrderService
import qualified Vayu.Services.Internal.Cart.Main as CartServices
import qualified Vayu.Services.Internal.Customer.Main as CustomerService
import qualified Vayu.Services.Internal.Shop.Main as Shop
import qualified Vayu.Services.Internal.Platform.Main as Platform

-- Import External Services (for third-party integrations)
import qualified Vayu.Services.External.Euler.Order as Euler
import qualified Vayu.Services.External.BreezeWallet.Main as BreezeWallet
import qualified Vayu.Services.External.Shopify.Order.Main as ShopifyOrder

-- Business workflow: Get Order by ID
getOrderWithId :: Text -> Bool -> FlowMonad.Flow Types.GetOrderResponse
getOrderWithId orderId isSyncOrderFlow = do
  -- Step 1: Log the request
  startTimestamp <- Logger.logProductAPIRequest "getOrderWithId" Logger.GET url params
  
  -- Step 2: Use Internal Service to fetch and update order
  (updatedOrder, mPaymentDetails) <- Utils.getOrderAndPaymentDetails orderId isSyncOrderFlow
  
  -- Step 3: Handle COD orders (update transaction)
  let isCodOrder = maybe False checkCOD mPaymentDetails
  when (isSuccess updatedOrder && isCodOrder) $
    void $ updateTransactionForOrder (updatedOrder ^. Order.id) mPaymentDetails
  
  -- Step 4: Create platform order if needed (calls External Service)
  result <- Utils.createPlatformOrder (updatedOrder, mPaymentDetails)
             >>= Utils.refundGCForFailed
             >>= Utils.mkGetOrderResponse isSyncOrderFlow
  
  -- Step 5: Log and return response
  Logger.logProductAPIResponse "getOrderWithId" startTimestamp result
```

### Product Layer Module Structure

```
src/Vayu/Product/Order/
├── Main.hs       # Exported API functions (business workflows)
├── Types.hs      # Domain-specific types
├── Utils.hs      # Helper functions for this domain
├── Constants.hs  # Module constants
└── Workflow.hs   # Complex workflow definitions
```

---

## 25. Layer 4: Internal Services

**Location**: `src/Vayu/Services/Internal/*/`

Internal services provide **reusable, pure business functions** that don't make external HTTP calls (except database/Redis).

```haskell
-- src/Vayu/Services/Internal/Order/Main.hs
module Vayu.Services.Internal.Order.Main (
  getAndUpdateNonTerminalOrder,
  processOrderOnFinish,
  getOrCreateOrderWithAddress,
  isOrderStatusTerminal,
  -- ... pure business logic
) where

-- Internal Services can import:
-- ✅ Other Internal Services
import qualified Vayu.Services.Internal.Cart.Main as CartService
import qualified Vayu.Services.Internal.Address.Main as AddressService
import qualified Vayu.Services.Internal.Customer.Main as Customer

-- ✅ External Services (for API calls)
import qualified Vayu.Services.External.Euler.Order as ExternalEulerOrder
import qualified Vayu.Services.External.WooCommerce.Session as WooCommerceExternal

-- ❌ NEVER import Product layer (would cause circular dependency!)

-- Pure business function
isOrderStatusTerminal :: Types.OrderStatus -> Bool
isOrderStatusTerminal orderStatus
  | orderStatus `elem` [SUCCESS, FAILED, AUTO_REFUNDED, PARTIALLY_PAID] = True
  | otherwise = False

-- DB operation with business logic
getOrCreateOrderWithAddress ::
  Cart.Cart ->
  Types.CustomerId ->
  Shop.Shop ->
  Types.AddressId ->
  Types.RecoverCart ->
  FlowMonad.Flow Order.Order
getOrCreateOrderWithAddress cart customerId shop selectedAddressId recover = do
  -- Query database
  OrderQueries.maybeFindOrderWithCartId cartId
    >>= maybe
      (createOrder cartId customerId shop selectedAddressId)  -- Create new
      (\order ->                                               -- Or update existing
        if recover && not (isOrderStatusTerminal $ order ^. status)
          then updateOrderStatusToPending order
          else findOrUpdateOrder order
      )
```

### Internal Service Module Structure

```
src/Vayu/Services/Internal/Order/
├── Main.hs       # Core business logic
├── Queries.hs    # Database operations (Beam queries)
├── Utils.hs      # Pure utility functions
├── Types.hs      # Service-specific types
└── Constants.hs  # Module constants
```

---

## 26. Layer 5: External Services

**Location**: `src/Vayu/Services/External/*/`

External services handle **third-party API integrations**.

```haskell
-- src/Vayu/Services/External/Shopify/Order/Main.hs
module Vayu.Services.External.Shopify.Order.Main (
  findOrCreateOrder,
  createBasket,
  fetchOrder,
  deleteOrder,
  -- ... Shopify API operations
) where

-- External Services CAN import Internal Services for:
-- ✅ Database queries
import qualified Vayu.Services.Internal.Offer.Main as Offer
import qualified Vayu.Services.Internal.Order.Queries as OrderQueries
import qualified Vayu.Services.Internal.Shipping.Main as Shipping

-- ❌ NEVER import Product layer!
-- ❌ Avoid importing other External Services (unless absolutely necessary)

-- Shopify API integration
findOrCreateOrder ::
  Customer.Customer ->
  Shop.Shop ->
  Cart.Cart ->
  Order.Order ->
  Maybe Address.Address ->
  FlowMonad.Flow (Either Types.Error Types.ShopifyOrderResponse)
findOrCreateOrder customer shop cart order shippingAddress = do
  -- Get shop configuration
  shopifySession <- ShopifySessionQueries.getShopifySession shopId
  
  -- Make HTTP call to Shopify
  response <- callServantAPI 
    (mkShopifyClient shopUrl authToken) 
    (createOrderRequest order cart)
  
  case response of
    Left err -> return $ Left $ mkError err
    Right shopifyOrder -> do
      -- Update our database with platform order ID
      OrderQueries.updatePlatformOrderId orderId (shopifyOrder ^. id)
      return $ Right shopifyOrder
```

### External Service Categories

```
src/Vayu/Services/External/
├── Shopify/         # Shopify Admin & Storefront API
├── WooCommerce/     # WooCommerce REST API
├── Magento/         # Magento 2 API
├── Vtex/            # VTEX Platform API
├── Euler/           # Juspay/Euler Payment Gateway
├── BreezeWallet/    # Breeze Wallet Service
├── Gupshup.hs       # WhatsApp messaging
├── Kaleyra/         # SMS/Voice provider
├── FCM/             # Firebase Cloud Messaging
├── AwsS3/           # S3 file storage
├── AwsSES/          # Email service
└── ... (60+ integrations)
```

---

## 27. Dependency Flow & Circular Dependency Prevention

### The Dependency DAG (Directed Acyclic Graph)

```
                    ┌─────────────────┐
                    │  Routes/Core.hs │
                    └────────┬────────┘
                             │ imports
                             ▼
                    ┌─────────────────┐
                    │  Product/*      │
                    └────────┬────────┘
                             │ imports
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │ Internal/* │   │ External/* │   │  Storage/* │
     └────────────┘   └────────────┘   └────────────┘
              │              │
              │    imports   │
              └──────►◄──────┘
```

### Rules to Prevent Circular Dependencies

| Layer | Can Import | Cannot Import |
|-------|------------|---------------|
| **Routes** | Product, Internal, External | - |
| **Product** | Internal, External | Routes |
| **Internal** | Other Internal, External, Storage | Product, Routes |
| **External** | Internal (limited), Storage | Product, Routes, Other External* |

*Exception: External services can import other External services for utility functions only.

### Example: How Internal Services Can Call External Services

```haskell
-- src/Vayu/Services/Internal/Platform/Order.hs
-- This is the "Platform" internal service that orchestrates external calls

module Vayu.Services.Internal.Platform.Order where

-- Import multiple External Services based on shop type
import qualified Vayu.Services.External.Shopify.Main as Shopify
import qualified Vayu.Services.External.WooCommerce.Main as WooCommerce
import qualified Vayu.Services.External.Magento.Main as Magento
import qualified Vayu.Services.External.Vtex.Main as Vtex

-- Platform-agnostic order creation
findOrCreateOrder :: Shop.Shop -> Order.Order -> ... -> Flow (Either Error OrderResponse)
findOrCreateOrder shop order ... =
  case shop ^. shopType of
    ShopType_SHOPIFY     -> Shopify.findOrCreateOrder ...
    ShopType_WOOCOMMERCE -> WooCommerce.createOrder ...
    ShopType_MAGENTO     -> Magento.createOrder ...
    ShopType_VTEX        -> Vtex.createOrder ...
```

---

## 28. Complete Request Flow Example: Get Order

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HTTP GET /order/ord_12345?sync=true                                         │
│ Headers: Authorization: Bearer <token>                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. API.hs - Route Matching                                                   │
│                                                                              │
│    "order" :> Capture "orderId" Text :> QueryParam "sync" Bool              │
│          :> Get '[JSON] GetOrderResponse                                     │
│                                                                              │
│    → Maps to: CoreRoutes.getOrderFromId                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Core.hs - Route Handler                                                   │
│                                                                              │
│    getOrderFromId orderId isSyncOrderFlow accessToken sessionId ... = do    │
│      flowWithTrace                                                           │
│        ( AuthMiddleware.validateS2SAuth accessToken READ ORDER              │
│            >>= handleBoolWithException                                       │
│            >> Order.getOrderWithId orderId (fromMaybe False isSyncOrderFlow)│
│        ) sessionId requestId shopUrl testingMode deviceId                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Product/Order/Main.hs - Business Workflow                                 │
│                                                                              │
│    getOrderWithId :: Text -> Bool -> Flow GetOrderResponse                  │
│    getOrderWithId orderId isSyncOrderFlow = do                              │
│      -- Log request                                                         │
│      startTimestamp <- Logger.logProductAPIRequest "getOrderWithId" ...     │
│                                                                              │
│      -- Fetch order (Internal Service)                                      │
│      (updatedOrder, mPaymentDetails) <- Utils.getOrderAndPaymentDetails ... │
│                                                                              │
│      -- Update transaction for COD                                          │
│      when isCodOrder $ updateTransactionForOrder ...                        │
│                                                                              │
│      -- Create platform order if needed (calls Shopify/WooCommerce)        │
│      result <- Utils.createPlatformOrder ...                                │
│                                                                              │
│      -- Log and return                                                      │
│      Logger.logProductAPIResponse "getOrderWithId" startTimestamp result    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│ 4a. Internal/Order/Main.hs        │   │ 4b. Internal/Platform/Order.hs    │
│                                   │   │                                   │
│ findWithId :: Text                │   │ findOrCreateOrder ::              │
│   -> Flow (Either Err Order)      │   │   Shop -> Order -> ...            │
│ findWithId orderId = do           │   │   -> Flow (Either Err Resp)       │
│   OrderQueries.findById ...       │   │ findOrCreateOrder shop ... =      │
│                                   │   │   case shop ^. type of            │
│                                   │   │     SHOPIFY -> Shopify.create     │
│                                   │   │     WOOCOMMERCE -> WC.create      │
└───────────────────────────────────┘   └───────────────────────────────────┘
                    │                               │
                    ▼                               ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│ 5a. Storage/Queries/Order.hs      │   │ 5b. External/Shopify/Order.hs     │
│                                   │   │                                   │
│ -- Beam database query            │   │ -- HTTP call to Shopify           │
│ findById orderId = do             │   │ create order shop cart = do       │
│   runDB $ select $ do             │   │   response <- callServantAPI      │
│     order <- all_ orderTable      │   │     shopifyClient                 │
│     guard_ (_id order ==. val)    │   │     (createOrderRequest ...)      │
│     return order                  │   │   return response                 │
└───────────────────────────────────┘   └───────────────────────────────────┘
```

---

## 29. Key Architecture Principles

### 29.1 Separation of Concerns

| Layer | Responsibility |
|-------|----------------|
| **Routes** | HTTP concerns only |
| **Product** | Business orchestration |
| **Internal** | Reusable business logic |
| **External** | Third-party integrations |

### 29.2 Dependency Injection via Flow

```haskell
-- All layers use FlowMonad.Flow
-- Environment (config, connections) implicit via Reader

getOrderWithId :: Text -> Bool -> FlowMonad.Flow Types.GetOrderResponse
getOrderWithId orderId isSyncOrderFlow = do
  config <- getConfig        -- Implicit via Reader
  shop <- getShopFromEnv     -- Implicit via Reader
  -- ... business logic
```

### 29.3 No Circular Dependencies

```
Strict import hierarchy:
Routes → Product → Internal/External

✅ Internal services CAN call External services
✅ External services CAN use Internal utilities
❌ Product/Internal CANNOT import Routes
❌ External CANNOT import Product
```

### 29.4 Platform Abstraction

```haskell
-- Internal/Platform/* abstracts Shopify, WooCommerce, etc.
-- Product layer is platform-agnostic

createOrder :: Order -> Flow (Either Error PlatformOrder)
createOrder order = do
  shop <- getShop
  Platform.findOrCreateOrder shop order  -- Delegates to correct platform
```

### 29.5 Error Handling Boundary

```haskell
-- Internal/External return Either Error Result
findOrder :: Text -> Flow (Either Error Order)

-- Product layer converts to HTTP errors
getOrder :: Text -> Flow GetOrderResponse
getOrder orderId = do
  result <- OrderService.findOrder orderId
  case result of
    Left err -> Logger.logAndThrow404 "getOrder" err
    Right order -> return $ mkResponse order

-- Routes layer handles HTTP error responses
-- (via ExceptT ServerError IO in the monad stack)
```

### 29.6 Architecture Benefits

| Benefit | How It's Achieved |
|---------|-------------------|
| **Pure functional style** | Side effects tracked by types (IO, Flow) |
| **Implicit dependency injection** | Env passed via Reader monad |
| **Testability** | Mock Env for unit tests |
| **Composability** | Build complex flows from simple ones |
| **Safe concurrency** | Structured threading with proper context |
| **Platform agnostic** | Internal/Platform abstracts external APIs |
| **Scalability** | Clear boundaries, easy to add new services |

---

## 30. Practice Exercises

```haskell
-- Exercise 1: Implement safeDiv chain
-- Divide 100 by a, then by b, then by c (return Nothing if any division by 0)
chainDiv :: Int -> Int -> Int -> Maybe Int
chainDiv a b c = do
  r1 <- safeDiv 100 a
  r2 <- safeDiv r1 b
  safeDiv r2 c

-- Exercise 2: Implement a simple parser monad
newtype Parser a = Parser { runParser :: String -> Maybe (a, String) }

-- Make it a Monad and implement:
-- char :: Char -> Parser Char (parse specific character)
-- satisfy :: (Char -> Bool) -> Parser Char (parse char matching predicate)

-- Exercise 3: Use List monad to generate all subsets of a list
subsets :: [a] -> [[a]]
subsets [] = [[]]
subsets (x:xs) = do
  subset <- subsets xs
  [subset, x:subset]

-- Exercise 4: Implement a Writer monad
newtype Writer w a = Writer { runWriter :: (a, w) }
-- Implement Functor, Applicative, Monad
-- Implement: tell :: w -> Writer w ()
```

---

## 31. Summary: The Complete Picture

### 31.1 The Type Class Hierarchy

```
Functor      -- Can map over: fmap / <$>
    |
    v
Applicative  -- Can apply wrapped functions: <*>, pure
    |
    v
Monad        -- Can chain dependent computations: >>=, return, do notation
```

### 22.2 Key Abstractions

| Abstraction | Key Operation | Use Case |
|-------------|---------------|----------|
| Functor | `fmap` | Transform value in context |
| Applicative | `<*>` | Combine independent effects |
| Monad | `>>=` | Chain dependent effects |

### 31.3 IO Monad Key Points

| Concept | Description |
|---------|-------------|
| `IO a` | A description of an action that produces `a` |
| `return`/`pure` | Wrap pure value in IO |
| `>>=` (bind) | Chain IO actions |
| `do` notation | Syntactic sugar for `>>=` |
| First-class | IO actions can be stored, passed, composed |
| Execution | Only `main` is executed by runtime |

### 31.4 FlowMonad Key Concepts (Vayu)

| Concept | Definition | Use in Vayu |
|---------|------------|-------------|
| **Monad** | Sequential computation with context | Chain operations with `do` notation |
| **ReaderT** | Monad transformer for implicit environment | Access `Env` without passing it |
| **Flow** | `ReaderT Env (EulerHS.Flow)` | All business logic runs in Flow |
| **Env** | Runtime context (config, connections, etc.) | Implicit dependency injection |
| **lift** | Bring inner monad action to outer | Access EulerHS from Flow |
| **ask** | Get the Reader environment | Access Env in any Flow function |
| **hoistServer** | Transform monad for all handlers | Connect Servant to Flow |

### 31.5 The Big Picture

```
┌────────────────────────────────────────────────────────────────┐
│                    APPLICATION STARTUP                          │
│  runVayuBackendApp creates FlowRuntime + Env + starts Warp     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    REQUEST HANDLING                             │
│  hoistServer transforms Flow to Handler for each request       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC (Flow)                        │
│  • Access config via getConfig                                 │
│  • Do DB operations via EulerHS                                │
│  • Spawn threads for async work                                │
│  • Return results                                              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    RESPONSE                                     │
│  Flow result → Servant serialization → HTTP response           │
└────────────────────────────────────────────────────────────────┘
```

This architecture provides:
- **Pure functional style** - Side effects are tracked by types
- **Implicit dependency injection** - Env passed via Reader
- **Testability** - Mock Env for unit tests
- **Composability** - Build complex flows from simple ones
- **Safe concurrency** - Structured threading with proper context

---

## Next: [Part 4 - Real World Haskell (Advanced IO, Concurrency, Error Handling)](./04-real-world.md)
