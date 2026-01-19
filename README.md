# Haskell-Learnings
📚 Haskell Learning Path: From basics to production-ready patterns with real-world examples

## Learning Path Overview

| Part | Topic | Key Concepts |
|------|-------|--------------|
| [01](./01-basics.md) | **Basics** | Syntax, functions, lists, pattern matching, higher-order functions |
| [02](./02-types.md) | **Types** | Type system, custom types, type classes, Maybe, Either |
| [03](./03-monads.md) | **Monads & Effects** | Functor → Applicative → Monad, IO deep dive, FlowMonad, Vayu Architecture |
| [04](./04-real-world.md) | **Real World** | Error handling, concurrency, lenses, Text/ByteString, Aeson, debugging |

---

## What's Covered

### Part 1: Basics
- Variables and basic types
- Functions and pattern matching
- Guards, where, let
- Lists and list comprehensions
- Higher-order functions (map, filter, fold)
- Currying and partial application

### Part 2: Type System
- Type signatures and type variables
- Type classes (Eq, Ord, Show, Functor)
- Custom types (type, newtype, data)
- Algebraic data types (sum/product)
- Maybe and Either
- Deriving instances

### Part 3: Monads & Effects
- Functor, Applicative, Monad hierarchy
- Do notation
- **IO Monad Deep Dive**
  - How IO works internally
  - Common IO operations
  - File handling
  - Exception handling
  - Concurrency (forkIO, MVar, STM, Async)
- Building custom monads (State, Reader)
- Monad transformers
- **FlowMonad (Vayu-specific)**
  - Monad transformer stack
  - Application startup sequence
  - Request lifecycle
  - Concurrency patterns
- **Vayu Architecture**
  - Three-tier service pattern
  - Generated API → Routes → Product → Internal/External
  - Dependency flow and circular prevention

### Part 4: Real World Haskell
- Error handling patterns
- Concurrency with Async
- Lenses for data manipulation
- **Lazy evaluation & strictness**
- **Common GHC extensions**
- **Text vs String, ByteString**
- **JSON with Aeson**
- **Debugging techniques**
- **Stack & Cabal basics**
- Testing with HSpec and QuickCheck

---

## Quick Reference

### Essential Operators

```haskell
-- Function application
f $ x           -- Same as f x, but $ has lowest precedence
f x y           -- Function application (left to right)

-- Function composition
f . g           -- (f . g) x = f (g x)

-- Functor
fmap f x        -- Apply f inside container
f <$> x         -- Same as fmap f x

-- Applicative
pure x          -- Wrap value in minimal context
f <*> x         -- Apply wrapped function to wrapped value

-- Monad
x >>= f         -- Bind: extract from x, apply f
x >> y          -- Sequence: run x, ignore result, run y
return x        -- Wrap value (same as pure)

-- Lens
x ^. lens       -- View/get
x & lens .~ v   -- Set
x & lens %~ f   -- Modify with function
```

### Type Signatures Cheat Sheet

```haskell
-- Reading type signatures
Int -> Int              -- Function: takes Int, returns Int
Int -> Int -> Int       -- Curried: takes Int, returns (Int -> Int)
[a] -> a                -- Polymorphic: list of any type, returns that type
Eq a => a -> a -> Bool  -- Constrained: 'a' must implement Eq

-- Common types
Maybe a                 -- Optional value: Nothing | Just a
Either e a              -- Error or success: Left e | Right a  
IO a                    -- Side-effectful computation returning a
[a]                     -- List of a
(a, b)                  -- Tuple of a and b
```

### Monad Hierarchy

```
        Functor     -- fmap / <$>
           ↓
      Applicative   -- <*>, pure
           ↓
         Monad      -- >>=, return, do-notation
```

---

## Vayu-Specific Patterns

### FlowMonad Usage

```haskell
import qualified FlowMonad
import qualified Vayu.Services.Logger.Logger as Logger

myFunction :: SomeInput -> FlowMonad.Flow (Either Error Result)
myFunction input = do
  -- Get config
  config <- getConfig
  
  -- Logging
  Logger.logInfo "MyModule:myFunction" ["input" .= input]
  
  -- Database query
  result <- SomeService.findById (input ^. GenAccessor.id)
  
  -- Return result
  case result of
    Nothing -> return $ Left NotFound
    Just r  -> return $ Right r
```

### Lens with Generated Accessors

```haskell
import qualified Vayu.Generated.Accessor as GenAccessor

-- Reading
let customerId = order ^. GenAccessor.customerId
let status = order ^. GenAccessor.status

-- Updating
let updatedOrder = order & GenAccessor.status .~ OrderStatus_SUCCESS
```

### Error Handling Pattern

```haskell
-- Internal service returns Either
createOrder :: OrderRequest -> Flow (Either Error Order)

-- Product layer handles both cases
handleRequest request = do
  result <- Order.createOrder request
  either 
    (\err -> Logger.logFailure tag (show err) >> throwErr400)
    (\order -> processOrder order)
    result
```

---

## How to Practice

1. **Start with Part 1**: Get comfortable with syntax and basic functions
2. **Build small programs**: Calculator, todo list, simple parsers
3. **Read Vayu code**: Study real patterns in the codebase
4. **Implement exercises**: Each part has practice exercises
5. **Use REPL**: `stack ghci` for interactive experimentation

---

## Quick Start Commands

```bash
# Start REPL
stack ghci

# In REPL
:t expression    -- Show type of expression
:i SomeType      -- Info about type/class
:l File.hs       -- Load file
:r               -- Reload

# Build project
stack build

# Run tests
stack test
```

---

## Common Gotchas

1. **No parentheses for function calls**: `add 3 5` not `add(3, 5)`
2. **Indentation matters**: Like Python, but stricter
3. **Immutability**: Variables can't be reassigned
4. **Lazy evaluation**: Values computed only when needed
5. **Type inference**: Often works, but explicit types are clearer

---

Created for the Vayu project. Happy learning! 🚀

---

## Document Statistics

| File | Sections | Topics |
|------|----------|--------|
| 01-basics.md | 6 | Syntax, Functions, Lists, HOFs |
| 02-types.md | 8 | Types, Type Classes, ADTs, Maybe/Either |
| 03-monads.md | 31 | Monads, IO, FlowMonad, Vayu Architecture |
| 04-real-world.md | 14 | IO, Concurrency, Lenses, Text, Aeson, Debugging |

**Total: ~4000+ lines of Haskell learning content**
