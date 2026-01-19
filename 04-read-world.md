# Haskell Learning Path - Part 4: Real World Haskell

This section covers practical aspects: IO, error handling, concurrency, and patterns used in production Haskell (like in Vayu).

---

## 1. IO in Depth

### 1.1 Understanding IO

```haskell
-- IO a = "A recipe for an action that, when executed, may perform 
--         side effects and produces a value of type 'a'"

-- Key insight: IO values are just descriptions of actions
-- They don't do anything until the runtime executes them

-- This is pure! It just returns a description
greet :: String -> IO ()
greet name = putStrLn ("Hello, " ++ name)

-- main is special: the runtime executes the IO action it returns
main :: IO ()
main = greet "World"
```

### 1.2 Common IO Operations

```haskell
-- Console
putStr     :: String -> IO ()
putStrLn   :: String -> IO ()
print      :: Show a => a -> IO ()
getLine    :: IO String
getChar    :: IO Char
getContents :: IO String  -- Lazy reading of all input

-- File operations
readFile   :: FilePath -> IO String
writeFile  :: FilePath -> String -> IO ()
appendFile :: FilePath -> String -> IO ()

-- Example: Read file and count lines
countLines :: FilePath -> IO Int
countLines path = do
  contents <- readFile path
  return $ length $ lines contents

-- Or more concise
countLines' :: FilePath -> IO Int
countLines' path = length . lines <$> readFile path
```

### 1.3 Lifting Pure Functions to IO

```haskell
-- Use fmap / <$> to apply pure functions to IO results
toUpper' :: IO String -> IO String
toUpper' = fmap (map toUpper)

-- Example
main :: IO ()
main = do
  line <- toUpper' getLine
  putStrLn line

-- More examples
getNumber :: IO Int
getNumber = read <$> getLine

addOne :: IO Int -> IO Int
addOne = fmap (+1)

-- Combining multiple IO values with Applicative
getName :: IO String
getName = do
  putStr "First name: "
  first <- getLine
  putStr "Last name: "
  last <- getLine
  return (first ++ " " ++ last)

-- With Applicative style
getName' :: IO String
getName' = (++) <$> (putStr "First: " >> getLine) 
               <*> (putStr " Last: " >> getLine)
```

---

## 2. Error Handling

### 2.1 Maybe for Optional Values

```haskell
-- When failure is expected and doesn't need explanation
lookup' :: Eq k => k -> [(k, v)] -> Maybe v
safeHead :: [a] -> Maybe a
safeDiv :: Int -> Int -> Maybe Int

-- Chaining with Maybe
processInput :: String -> Maybe Int
processInput input = do
  number <- readMaybe input
  guard (number > 0)
  return (number * 2)
```

### 2.2 Either for Recoverable Errors

```haskell
-- When you need to know WHY it failed
data AppError
  = NotFound String
  | ValidationError String  
  | DatabaseError String
  | NetworkError String
  deriving (Show, Eq)

-- Either e a: Left e = error, Right a = success
type Result a = Either AppError a

findUser :: Int -> Result User
findUser uid = 
  case lookup uid userDb of
    Nothing -> Left (NotFound $ "User " ++ show uid ++ " not found")
    Just u  -> Right u

-- Chaining Either computations
processOrder :: Int -> Int -> Result String
processOrder userId orderId = do
  user  <- findUser userId
  order <- findOrder orderId
  guard (orderUserId order == userId) `orError` 
    ValidationError "Order doesn't belong to user"
  return $ "Processing " ++ show order ++ " for " ++ userName user
  where
    orError :: Maybe a -> AppError -> Either AppError a
    orError Nothing  e = Left e
    orError (Just x) _ = Right x
```

### 2.3 Exceptions (for Unexpected Errors)

```haskell
import Control.Exception

-- Catching exceptions
safeReadFile :: FilePath -> IO (Either IOError String)
safeReadFile path = try (readFile path)

-- Throwing exceptions
validatePositive :: Int -> IO Int
validatePositive n
  | n <= 0    = throwIO $ userError "Must be positive"
  | otherwise = return n

-- Catching specific exceptions
handleFileError :: FilePath -> IO String
handleFileError path = 
  readFile path `catch` \(e :: IOError) ->
    return $ "Error reading file: " ++ show e

-- bracket for resource management (like try-finally)
withFile' :: FilePath -> (Handle -> IO a) -> IO a
withFile' path action = 
  bracket 
    (openFile path ReadMode)   -- Acquire resource
    hClose                      -- Release resource (always runs)
    action                      -- Use resource

-- Example
processFile :: FilePath -> IO String
processFile path = withFile' path $ \handle -> do
  contents <- hGetContents handle
  return $ process contents
```

### 2.4 Best Practices

```haskell
-- 1. Use Maybe for "not found" / optional
-- 2. Use Either for domain errors with context
-- 3. Use exceptions only for truly unexpected errors
-- 4. Always use bracket for resource cleanup

-- Pattern: Convert exceptions to Either at boundaries
safeIO :: IO a -> IO (Either String a)
safeIO action = (Right <$> action) `catch` 
  \(e :: SomeException) -> return $ Left (show e)
```

---

## 3. Concurrency

### 3.1 Basic Threading

```haskell
import Control.Concurrent

-- Fork a new thread
main :: IO ()
main = do
  forkIO $ putStrLn "Hello from thread 1"
  forkIO $ putStrLn "Hello from thread 2"
  threadDelay 1000000  -- Wait 1 second (in microseconds)
  putStrLn "Main thread done"

-- Thread IDs
main' :: IO ()
main' = do
  tid <- forkIO $ forever $ do
    putStrLn "Working..."
    threadDelay 500000
  
  threadDelay 2000000
  killThread tid  -- Stop the worker
  putStrLn "Killed worker thread"
```

### 3.2 MVars (Mutable Variables)

```haskell
import Control.Concurrent.MVar

-- MVar a: A synchronized mutable variable
-- Can be empty or contain one value

main :: IO ()
main = do
  -- Create empty MVar
  box <- newEmptyMVar
  
  -- Fork thread to put value
  forkIO $ do
    threadDelay 1000000
    putMVar box "Hello!"
  
  -- This blocks until box has a value
  value <- takeMVar box
  putStrLn value

-- Common pattern: MVar as mutex
type Lock = MVar ()

withLock :: Lock -> IO a -> IO a
withLock lock action = do
  takeMVar lock      -- Acquire lock
  result <- action
  putMVar lock ()    -- Release lock
  return result
```

### 3.3 STM (Software Transactional Memory)

```haskell
import Control.Concurrent.STM

-- TVar: Transactional variable
-- All reads/writes in 'atomically' block are atomic

-- Transfer money between accounts atomically
transfer :: TVar Int -> TVar Int -> Int -> STM ()
transfer from to amount = do
  fromBalance <- readTVar from
  if fromBalance < amount
    then retry  -- Block until condition might change
    else do
      writeTVar from (fromBalance - amount)
      toBalance <- readTVar to
      writeTVar to (toBalance + amount)

main :: IO ()
main = do
  account1 <- newTVarIO 100
  account2 <- newTVarIO 50
  
  -- Run transfer atomically
  atomically $ transfer account1 account2 30
  
  -- Check balances
  balance1 <- readTVarIO account1
  balance2 <- readTVarIO account2
  print (balance1, balance2)  -- (70, 80)
```

### 3.4 Async Library (Recommended)

```haskell
import Control.Concurrent.Async

-- Run actions concurrently and wait for both
main :: IO ()
main = do
  (result1, result2) <- concurrently
    (fetchUrl "http://example.com/1")
    (fetchUrl "http://example.com/2")
  print (result1, result2)

-- Race: return first to complete
fastest :: IO String
fastest = race
  (threadDelay 1000000 >> return "slow")
  (threadDelay 500000 >> return "fast")
  >>= either return return  -- "fast"

-- Map concurrently over list
fetchAll :: [String] -> IO [Response]
fetchAll urls = mapConcurrently fetchUrl urls

-- With timeout
withTimeout :: Int -> IO a -> IO (Maybe a)
withTimeout microseconds action = 
  race (threadDelay microseconds) action >>= \case
    Left _  -> return Nothing
    Right a -> return (Just a)
```

---

## 4. Lenses (Used Heavily in Vayu)

### 4.1 The Problem

```haskell
data Address = Address { city :: String, street :: String }
data Person = Person { name :: String, address :: Address }

-- Updating nested fields is painful
updateCity :: String -> Person -> Person
updateCity newCity person = person 
  { address = (address person) { city = newCity } }

-- Gets worse with deeper nesting!
```

### 4.2 Lens Basics

```haskell
{-# LANGUAGE TemplateHaskell #-}
import Control.Lens

-- Using Template Haskell to generate lenses
data Address = Address 
  { _city :: String
  , _street :: String 
  } deriving (Show)
makeLenses ''Address

data Person = Person 
  { _name :: String
  , _address :: Address 
  } deriving (Show)
makeLenses ''Person

-- Now we have:
-- city :: Lens' Address String
-- street :: Lens' Address String
-- name :: Lens' Person String
-- address :: Lens' Person Address

-- View (get)
person = Person "Alice" (Address "Mumbai" "MG Road")
person ^. name                     -- "Alice"
person ^. address . city           -- "Mumbai"

-- Set
person & name .~ "Bob"             -- Person with name = "Bob"
person & address . city .~ "Delhi" -- Update nested city

-- Modify
person & name %~ map toUpper       -- Person with name = "ALICE"
person & address . city %~ (++ "!")-- Person with city = "Mumbai!"

-- Combine operations
person & name .~ "Bob" 
       & address . city .~ "Delhi"
```

### 4.3 Common Lens Operations

```haskell
-- ^. : view (get value)
-- .~ : set (replace value)
-- %~ : over (modify with function)
-- +~ : add to numeric field
-- -~ : subtract from numeric field
-- <>~ : append to monoid field
-- ?~ : set to Just value
-- ^? : safe view (returns Maybe)
-- ^.. : get all values (for traversals)

data Stats = Stats { _score :: Int, _tags :: [String] }
makeLenses ''Stats

stats = Stats 100 ["easy"]

stats ^. score           -- 100
stats & score +~ 10      -- Stats 110 ["easy"]
stats & tags <>~ ["fun"] -- Stats 100 ["easy", "fun"]

-- Optional fields
data User = User { _userName :: String, _userAge :: Maybe Int }
makeLenses ''User

user = User "Alice" Nothing
user & userAge ?~ 25     -- User "Alice" (Just 25)
user ^? userAge . _Just  -- Nothing (safe access)
```

### 4.4 Lens in Vayu Pattern

```haskell
-- Vayu uses generated accessors like this:
import qualified Vayu.Generated.Accessor as GenAccessor

-- Getting values
customerId = order ^. GenAccessor.customerId
shopId = customer ^. GenAccessor.shopId

-- Setting values
updatedOrder = order & GenAccessor.status .~ OrderStatus_SUCCESS

-- Pattern matching with lens
case order ^. GenAccessor.paymentMethod of
  Just Types.PaymentMethod_COD -> handleCOD
  Just Types.PaymentMethod_PREPAID -> handlePrepaid
  Nothing -> handleUnknown
```

---

## 5. Monad Transformers in Practice

### 5.1 ReaderT Pattern

```haskell
import Control.Monad.Reader

-- Application environment
data AppEnv = AppEnv
  { envConfig   :: Config
  , envDbPool   :: Pool Connection
  , envLogger   :: Logger
  }

-- App monad: Reader + IO
type App = ReaderT AppEnv IO

-- Running the app
runApp :: AppEnv -> App a -> IO a
runApp env action = runReaderT action env

-- Using environment
getConfig :: App Config
getConfig = asks envConfig

logInfo :: String -> App ()
logInfo msg = do
  logger <- asks envLogger
  liftIO $ writeLog logger msg

-- Database query
queryUsers :: App [User]
queryUsers = do
  pool <- asks envDbPool
  liftIO $ withPool pool $ \conn ->
    query conn "SELECT * FROM users"
```

### 5.2 ExceptT for Error Handling

```haskell
import Control.Monad.Except

type AppM = ExceptT AppError (ReaderT AppEnv IO)

-- Throwing errors
findUser :: Int -> AppM User
findUser uid = do
  mUser <- lift $ queryUserById uid
  case mUser of
    Nothing -> throwError (NotFound "User not found")
    Just u  -> return u

-- Catching errors
processRequest :: Request -> AppM Response
processRequest req = 
  handleRequest req `catchError` \err ->
    return (errorResponse err)

-- Running the stack
runAppM :: AppEnv -> AppM a -> IO (Either AppError a)
runAppM env action = 
  runReaderT (runExceptT action) env
```

### 5.3 The Vayu FlowMonad

```haskell
-- Vayu uses a custom monad called Flow
-- It combines: Reader (config), State, IO, Error handling, Logging

-- Example from Vayu
processOrder :: OrderRequest -> FlowMonad.Flow (Either Error Order)
processOrder request = do
  -- Access config
  config <- getConfig
  
  -- Logging
  Logger.logInfo "Order:create" ["orderId" .= orderId]
  
  -- Database operations
  mUser <- findUserById (request ^. userId)
  
  -- Error handling
  case mUser of
    Nothing -> return $ Left UserNotFound
    Just user -> do
      order <- createOrder request user
      return $ Right order
```

---

## 6. Common Patterns in Production Haskell

### 6.1 Smart Constructors

```haskell
-- Don't export data constructors, export smart constructors
module Email (Email, mkEmail, unEmail) where

newtype Email = Email String
  deriving (Show, Eq)

-- Smart constructor validates
mkEmail :: String -> Maybe Email
mkEmail s
  | '@' `elem` s && '.' `elem` s = Just (Email s)
  | otherwise = Nothing

-- Accessor
unEmail :: Email -> String
unEmail (Email s) = s

-- Now invalid emails can't be created!
```

### 6.2 Phantom Types

```haskell
-- Type-level tags for extra safety
data Validated
data Unvalidated

newtype UserId (a :: *) = UserId Int
  deriving (Show, Eq)

-- Only accept validated user IDs
processUser :: UserId Validated -> IO ()
processUser (UserId uid) = putStrLn $ "Processing user " ++ show uid

-- Validation produces Validated type
validateUserId :: UserId Unvalidated -> Maybe (UserId Validated)
validateUserId (UserId uid)
  | uid > 0   = Just (UserId uid)
  | otherwise = Nothing

-- Can't call processUser with unvalidated ID!
```

### 6.3 Type-Safe Builders

```haskell
{-# LANGUAGE DataKinds #-}

-- Track required fields at type level
data Required
data Optional

data RequestBuilder (name :: *) (age :: *) = RequestBuilder
  { rbName :: Maybe String
  , rbAge  :: Maybe Int
  }

emptyBuilder :: RequestBuilder Optional Optional
emptyBuilder = RequestBuilder Nothing Nothing

setName :: String -> RequestBuilder n a -> RequestBuilder Required a
setName name rb = rb { rbName = Just name }

setAge :: Int -> RequestBuilder n a -> RequestBuilder n Required
setAge age rb = rb { rbAge = Just age }

-- Can only build if both fields are set
build :: RequestBuilder Required Required -> Request
build rb = Request (fromJust $ rbName rb) (fromJust $ rbAge rb)

-- This pattern ensures you can't forget required fields
```

---

## 7. Testing

### 7.1 Unit Testing with HSpec

```haskell
-- test/Spec.hs
import Test.Hspec

main :: IO ()
main = hspec $ do
  describe "safeHead" $ do
    it "returns Just for non-empty list" $ do
      safeHead [1,2,3] `shouldBe` Just 1
    
    it "returns Nothing for empty list" $ do
      safeHead ([] :: [Int]) `shouldBe` Nothing

  describe "Calculator" $ do
    it "adds numbers correctly" $ do
      add 2 3 `shouldBe` 5
    
    it "handles negative numbers" $ do
      add (-1) 5 `shouldBe` 4
```

### 7.2 Property-Based Testing with QuickCheck

```haskell
import Test.QuickCheck

-- Properties that should hold for all inputs
prop_reverseReverse :: [Int] -> Bool
prop_reverseReverse xs = reverse (reverse xs) == xs

prop_lengthAppend :: [Int] -> [Int] -> Bool
prop_lengthAppend xs ys = 
  length (xs ++ ys) == length xs + length ys

-- Run with: quickCheck prop_reverseReverse
-- QuickCheck generates random test cases automatically!

-- Custom generators
newtype PositiveInt = PositiveInt Int deriving (Show)

instance Arbitrary PositiveInt where
  arbitrary = PositiveInt . abs <$> arbitrary

prop_sqrtPositive :: PositiveInt -> Bool
prop_sqrtPositive (PositiveInt n) = sqrt (fromIntegral n) >= 0
```

---

## 8. Lazy Evaluation & Strictness

### 8.1 Understanding Laziness

```haskell
-- Haskell is lazy: values computed only when needed

-- This is fine (infinite list!)
naturals :: [Int]
naturals = [1..]

-- We can take from it
take 5 naturals  -- [1,2,3,4,5]

-- Lazy evaluation means expressions are "thunks" (unevaluated)
-- x = 1 + 2  -- x is a thunk, not 3 yet
-- When we need x, it gets "forced" to 3

-- Benefits:
-- 1. Infinite data structures
-- 2. Only compute what's needed
-- 3. Composable pipelines

-- Example: First 10 primes
primes = sieve [2..]
  where sieve (p:xs) = p : sieve [x | x <- xs, x `mod` p /= 0]

take 10 primes  -- [2,3,5,7,11,13,17,19,23,29]
```

### 8.2 Problems with Laziness

```haskell
-- Space leaks: Thunks accumulate in memory

-- Bad: Builds up huge chain of thunks
badSum :: [Int] -> Int
badSum = foldl (+) 0

badSum [1..1000000]  
-- Creates: ((((0+1)+2)+3)+...+1000000)
-- All those thunks stay in memory!

-- Good: Force evaluation at each step
import Data.List (foldl')

goodSum :: [Int] -> Int
goodSum = foldl' (+) 0  -- Strict left fold

-- foldl' forces the accumulator at each step
```

### 8.3 Forcing Evaluation

```haskell
import Control.DeepSeq

-- seq: Evaluate to WHNF (Weak Head Normal Form)
-- Forces outermost constructor only
x `seq` y  -- Evaluate x, then return y

strictSum :: Int -> Int -> Int
strictSum a b = a `seq` b `seq` (a + b)

-- deepseq: Evaluate completely (Normal Form)
-- Forces entire structure
x `deepseq` y  -- Fully evaluate x, then return y

-- Bang patterns (with BangPatterns extension)
{-# LANGUAGE BangPatterns #-}

strictSum' :: Int -> Int -> Int
strictSum' !a !b = a + b  -- ! forces evaluation

-- Strict data fields
data Point = Point !Double !Double  -- Fields evaluated when Point created
```

### 8.4 When to Use Strictness

```haskell
-- Use strict when:
-- 1. Accumulating values (folds)
-- 2. Numeric computations
-- 3. Data structures that shouldn't hold thunks

-- Use lazy when:
-- 1. Working with potentially infinite data
-- 2. Early termination is needed
-- 3. Conditional evaluation

-- Vayu pattern: Strict fields in data types
data Order = Order
  { _orderId     :: !Text      -- Strict
  , _orderAmount :: !Double    -- Strict
  , _orderItems  :: [Item]     -- Lazy (might not need all items)
  }
```

---

## 9. Common GHC Extensions

### 9.1 Essential Extensions

```haskell
{-# LANGUAGE OverloadedStrings #-}
-- String literals can be Text, ByteString, etc.
import Data.Text (Text)

greeting :: Text
greeting = "Hello"  -- No need for T.pack "Hello"

{-# LANGUAGE RecordWildCards #-}
-- Bring all record fields into scope
data Config = Config { host :: String, port :: Int }

showConfig :: Config -> String
showConfig Config{..} = host ++ ":" ++ show port
-- host and port are in scope

{-# LANGUAGE LambdaCase #-}
-- Pattern match directly on lambda argument
handleResult = \case
  Nothing -> "No result"
  Just x  -> "Got: " ++ show x

{-# LANGUAGE TupleSections #-}
-- Partial tuple application
pairs = map (,True) [1,2,3]  -- [(1,True), (2,True), (3,True)]
```

### 9.2 Type-Level Extensions

```haskell
{-# LANGUAGE TypeApplications #-}
-- Explicitly specify type arguments
read @Int "42"  -- 42 :: Int
show @Bool True -- "True"

{-# LANGUAGE ScopedTypeVariables #-}
-- Use type variables in where clauses
example :: forall a. Show a => a -> String
example x = show (x :: a)  -- Can reference 'a' in body

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
-- More flexible type class constraints and instances

{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
-- Control how instances are derived
data User = User { name :: Text, age :: Int }
  deriving stock (Show, Eq, Generic)     -- Standard deriving
  deriving anyclass (ToJSON, FromJSON)   -- Via Generic
```

### 9.3 Vayu Common Extensions

```haskell
-- Typically enabled in Vayu files:
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ViewPatterns #-}

-- ViewPatterns example
import qualified Data.Text as T

processText :: Text -> String
processText (T.unpack -> s) = s ++ "!"  -- Pattern with function
```

---

## 10. Working with Text and ByteString

### 10.1 Text vs String

```haskell
-- String = [Char]  -- Linked list, inefficient!
-- Text = packed UTF-16 -- Efficient for unicode text

import qualified Data.Text as T
import qualified Data.Text.IO as TIO

-- Converting
textFromString :: String -> T.Text
textFromString = T.pack

stringFromText :: T.Text -> String
stringFromText = T.unpack

-- With OverloadedStrings, use Text directly
{-# LANGUAGE OverloadedStrings #-}

greeting :: T.Text
greeting = "Hello, World!"

-- Common operations
T.length "hello"          -- 5
T.toUpper "hello"         -- "HELLO"
T.append "a" "b"          -- "ab"
"a" <> "b"                -- "ab" (Monoid)
T.intercalate ", " ["a","b","c"]  -- "a, b, c"
T.splitOn "," "a,b,c"     -- ["a", "b", "c"]

-- IO with Text
TIO.readFile "file.txt"   -- IO Text
TIO.writeFile "out.txt" content
```

### 10.2 ByteString for Binary Data

```haskell
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Char8 as BC  -- ASCII

-- Strict ByteString: entire content in memory
readBinary :: FilePath -> IO BS.ByteString
readBinary = BS.readFile

-- Lazy ByteString: streamed in chunks
readLargeBinary :: FilePath -> IO BL.ByteString  
readLargeBinary = BL.readFile

-- Converting Text <-> ByteString
import Data.Text.Encoding (encodeUtf8, decodeUtf8)

textToBS :: T.Text -> BS.ByteString
textToBS = encodeUtf8

bsToText :: BS.ByteString -> T.Text
bsToText = decodeUtf8  -- Can throw on invalid UTF-8

bsToTextSafe :: BS.ByteString -> Either String T.Text
bsToTextSafe = decodeUtf8'
```

---

## 11. JSON with Aeson

### 11.1 Basic Usage

```haskell
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

import Data.Aeson
import GHC.Generics

-- Automatic JSON instances
data User = User
  { userName :: Text
  , userAge  :: Int
  , userEmail :: Maybe Text
  }
  deriving (Show, Generic, ToJSON, FromJSON)

-- Encoding
userJson :: ByteString
userJson = encode (User "Alice" 25 (Just "alice@example.com"))
-- {"userName":"Alice","userAge":25,"userEmail":"alice@example.com"}

-- Decoding
parseUser :: ByteString -> Maybe User
parseUser = decode

-- With Either for error messages
parseUserE :: ByteString -> Either String User
parseUserE = eitherDecode
```

### 11.2 Custom JSON Instances

```haskell
import Data.Aeson

data Status = Active | Inactive | Pending
  deriving (Show, Eq)

instance ToJSON Status where
  toJSON Active   = String "active"
  toJSON Inactive = String "inactive"
  toJSON Pending  = String "pending"

instance FromJSON Status where
  parseJSON = withText "Status" $ \t ->
    case t of
      "active"   -> pure Active
      "inactive" -> pure Inactive
      "pending"  -> pure Pending
      _          -> fail "Invalid status"

-- Custom field names
data Order = Order
  { orderId :: Text
  , orderTotal :: Double
  }
  deriving (Show, Generic)

instance ToJSON Order where
  toJSON = genericToJSON defaultOptions
    { fieldLabelModifier = camelTo2 '_' . drop 5  -- Remove "order" prefix
    }
-- {"id":"123","total":99.99}
```

### 11.3 Working with Dynamic JSON

```haskell
import Data.Aeson
import qualified Data.Aeson.KeyMap as KM

-- Parse unknown JSON structure
parseAny :: ByteString -> Maybe Value
parseAny = decode

-- Value can be:
-- Object (KeyMap Value)
-- Array (Vector Value)
-- String Text
-- Number Scientific
-- Bool Bool
-- Null

-- Extract fields
extractName :: Value -> Maybe Text
extractName (Object o) = case KM.lookup "name" o of
  Just (String n) -> Just n
  _ -> Nothing
extractName _ = Nothing

-- Lens-Aeson for easy access
import Data.Aeson.Lens

json ^? key "user" . key "name" . _String  -- Maybe Text
json ^? key "items" . _Array               -- Maybe (Vector Value)
json ^? key "count" . _Integer             -- Maybe Integer
```

---

## 12. Debugging Techniques

### 12.1 Debug.Trace

```haskell
import Debug.Trace

-- Print and return value
traced :: Int -> Int
traced x = trace ("x = " ++ show x) (x + 1)

-- Print in a pipeline
result = [1,2,3]
  & map (*2)
  & traceShowId  -- Prints and returns: [2,4,6]
  & filter (>3)

-- Conditional tracing
debugMode = True

debugLog :: String -> a -> a
debugLog msg = if debugMode then trace msg else id

-- In IO
import Debug.Trace (traceIO)

main = do
  traceIO "Starting..."
  result <- computation
  traceIO $ "Result: " ++ show result
```

### 12.2 GHCi Debugging

```haskell
-- In GHCi:

:t expression        -- Show type
:i TypeOrClass       -- Info about type/class
:k TypeConstructor   -- Kind of type
:l Module.hs         -- Load file
:r                   -- Reload
:set -fwarn-all      -- Enable all warnings

-- Breakpoints
:break ModuleName lineNumber
:continue
:step
:list                -- Show code around current point
:show bindings       -- Show variables in scope

-- Evaluate expressions
> let x = [1..10]
> x
[1,2,3,4,5,6,7,8,9,10]
> :sprint x          -- Show evaluated vs thunks
x = [1,2,3,4,5,6,7,8,9,10]
```

### 12.3 Common Debugging Patterns

```haskell
-- Log function inputs/outputs
debugFn :: (Show a, Show b) => String -> (a -> b) -> a -> b
debugFn name f x = 
  trace (name ++ " input: " ++ show x ++ " output: " ++ show result) result
  where result = f x

-- Use in pipeline
process = debugFn "step1" step1
        . debugFn "step2" step2
        . debugFn "step3" step3

-- Vayu logging pattern
import qualified Vayu.Services.Logger.Logger as Logger

myFunction :: Input -> FlowMonad.Flow Output
myFunction input = do
  Logger.logDebug "MyModule:myFunction" ["input" .= input]
  result <- doSomething input
  Logger.logDebug "MyModule:myFunction" ["result" .= result]
  return result
```

---

## 13. Build Tools: Stack & Cabal

### 13.1 Stack Commands

```bash
# Project setup
stack new my-project    # Create new project
stack setup             # Install GHC

# Building
stack build             # Build project
stack build --fast      # Fast build (no optimization)
stack build --pedantic  # Treat warnings as errors

# Running
stack exec my-app       # Run executable
stack run               # Build and run (newer Stack)

# Testing
stack test              # Run all tests
stack test --coverage   # With coverage report

# REPL
stack ghci              # Start GHCi with project loaded
stack ghci --test       # Include test modules

# Dependencies
stack ls dependencies   # List all dependencies
```

### 13.2 Package.yaml (Stack)

```yaml
name: my-project
version: 0.1.0.0

dependencies:
  - base >= 4.7 && < 5
  - text
  - aeson
  - lens
  - mtl

library:
  source-dirs: src
  exposed-modules:
    - MyModule
    - MyModule.Internal

executables:
  my-app:
    main: Main.hs
    source-dirs: app
    dependencies:
      - my-project

tests:
  my-test:
    main: Spec.hs
    source-dirs: test
    dependencies:
      - my-project
      - hspec
      - QuickCheck

default-extensions:
  - OverloadedStrings
  - LambdaCase
  - RecordWildCards
```

### 13.3 Useful Stack/GHCi Settings

```bash
# ~/.ghci (GHCi config)
:set prompt "λ> "
:set +m                  # Multi-line input
:set -fwarn-unused-binds
:set -fwarn-unused-imports

# View generated Core
stack build --ghc-options="-ddump-simpl"

# Profile memory usage
stack build --profile
stack exec -- my-app +RTS -h -p
```

---

## 14. Summary: Haskell in Production

| Concept | Use Case | Vayu Example |
|---------|----------|--------------|
| `Maybe` | Optional values | `findUser :: UserId -> Maybe User` |
| `Either` | Error handling with context | `Result a = Either AppError a` |
| `ReaderT` | Dependency injection | Config, DB pool access |
| Lenses | Nested data access/update | `order ^. GenAccessor.status` |
| STM | Concurrent state | Rate limiting, caching |
| Async | Parallel operations | Fetching from multiple APIs |
| `Text` | Efficient string handling | All text data in Vayu |
| Aeson | JSON serialization | API request/response |
| Strictness | Performance optimization | Strict fields in data types |

---

## Next Steps

1. **Practice**: Implement exercises from each part
2. **Read Code**: Study Vayu codebase patterns
3. **Build Something**: Create a small project
4. **Advanced Topics**: GADTs, Type Families, Free Monads

## Recommended Resources

- [Learn You a Haskell](http://learnyouahaskell.com/) - Free online book
- [Real World Haskell](http://book.realworldhaskell.org/) - Practical applications
- [Haskell Wiki](https://wiki.haskell.org/) - Reference
- [Typeclassopedia](https://wiki.haskell.org/Typeclassopedia) - Deep dive into type classes
